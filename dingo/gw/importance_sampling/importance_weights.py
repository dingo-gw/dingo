"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""

import logging
from pathlib import Path

from os import rename, makedirs
from os.path import join, isfile, exists

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dingo.core.posterior_models import NormalizingFlowPosteriorModel
from dingo.gw.result import Result
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.importance_sampling.diagnostics import plot_diagnostics

log = logging.getLogger(__name__)
logging.captureWarnings(True)


def _resolve_importance_input_paths(settings: dict) -> None:
    parameter_samples = settings.get("parameter_samples")
    if parameter_samples is None:
        parameter_samples = (
            settings.get("nde", {}).get("data", {}).get("parameter_samples")
        )
    if parameter_samples is None:
        raise KeyError("Missing required setting parameter_samples.")

    parameter_samples = to_absolute_path(parameter_samples)
    settings["parameter_samples"] = parameter_samples
    if "nde" in settings:
        settings["nde"].setdefault("data", {})
        settings["nde"]["data"]["parameter_samples"] = parameter_samples

    calibration = settings.get("calibration_marginalization")
    if calibration and "calibration_envelope" in calibration:
        calibration["calibration_envelope"] = {
            ifo: to_absolute_path(path)
            for ifo, path in calibration["calibration_envelope"].items()
        }


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="importance_weights",
)
def main(cfg: DictConfig):
    settings = OmegaConf.to_container(cfg, resolve=True)
    outdir = settings.pop("outdir")
    if outdir is None:
        outdir = "."
    makedirs(outdir, exist_ok=True)
    _resolve_importance_input_paths(settings)

    result = Result(file_name=settings["parameter_samples"])
    metadata = result.settings
    samples = result.samples
    # for time marginalization, we drop geocent time from the samples
    inference_parameters = metadata["train_settings"]["data"][
        "inference_parameters"
    ].copy()
    time_marginalization_kwargs = settings.get("time_marginalization", None)
    time_marginalization = time_marginalization_kwargs is not None
    phase_marginalization_kwargs = settings.get("phase_marginalization", None)
    phase_marginalization = phase_marginalization_kwargs is not None
    calibration_marginalization_kwargs = settings.get(
        "calibration_marginalization", None
    )
    synthetic_phase_kwargs = settings.get("synthetic_phase", None)
    synthetic_phase = synthetic_phase_kwargs is not None
    # if sum([time_marginalization, phase_marginalization, synthetic_phase]) > 1:
    #    raise NotImplementedError(
    #        "Only one of time_marginalization, phase_marginalization and"
    #        "synthetic_phase can be set to True."
    #    )
    if time_marginalization and "geocent_time" in samples:
        if "geocent_time" in inference_parameters:
            samples.drop("geocent_time", axis=1, inplace=True)
            inference_parameters.remove("geocent_time")
    if phase_marginalization or synthetic_phase:
        if "phase" in inference_parameters:
            samples.drop("phase", axis=1, inplace=True)
            inference_parameters.remove("phase")
    if "nde" in settings:
        settings["nde"]["data"]["inference_parameters"] = inference_parameters
        settings["nde"]["data"]["parameters"] = inference_parameters
        # TODO: train_unconditional_density_estimator should not accept
        #  settings["data"]["parameters"], such that the line above can be removed.

    # Step 1: Build proposal distribution.
    #
    # We use the dingo posterior as our proposal distribution. We need to be able to
    # sample from, and evaluate this distribution. Here, we therefore train an
    # unconditional neural density estimator (nde) to recover the posterior density from
    # dingo samples. This typically required, since one loses the tractable likelihood
    # when using GNPE for dingo. This is not a big deal though, since we can cheaply
    # sample from the dingo posterior, such that one can easily train an unconditional
    # nde to recover the posterior density.

    if "log_prob" not in samples.columns:
        # Use GPS time as name for now.
        event_name = str(result.event_metadata["time_event"])
        nde_name = settings["nde"].get("path") or join(outdir, f"nde-{event_name}.pt")
        if isfile(nde_name):
            log.info(f"Loading nde at {nde_name} for event {event_name}.")
            nde = NormalizingFlowPosteriorModel(
                model_filename=nde_name,
                device=settings["nde"]["training"]["device"],
                load_training_info=False,
            )
        else:
            log.info(f"Training new nde for event {event_name}.")
            nde = result.train_unconditional_flow(
                inference_parameters,
                settings["nde"],
                train_dir=outdir,
            )
            log.info(f"Renaming trained nde model to {nde_name}.")
            rename(join(outdir, "model_latest.pt"), nde_name)

        # Step 1a: Sample from proposal.
        nde_sampler = GWSampler(model=nde)
        nde_sampler.run_sampler(num_samples=settings["num_samples"])
        result = nde_sampler.to_result()

    # else:
    #     nde_sampler = GWSamplerUnconditional(
    #         result=result,
    #         synthetic_phase_kwargs=synthetic_phase_kwargs,
    #     )
    #
    # # Step 2: Sample from proposal.
    #
    # print(f'Generating {settings["num_samples"]} samples from proposal distribution.')
    # nde_sampler.run_sampler(num_samples=settings["num_samples"])

    # Step 2: Importance sample.
    #
    # Our target distribution is the posterior p(theta|d) = p(d|theta) * p(theta) / p(
    # d). For importance sampling, we need to evaluate the *unnormalized* posterior,
    # so we only need the likelihood p(d|theta) and the prior p(theta), but not the
    # evidence p(d).
    #
    # Sample from proposal distribution q(theta|d) and reweight the samples theta_i with
    #
    #       w_i = p(theta_i|d) / q(theta_i|d)
    #
    # to obtain weighted samples from the proposal distribution.

    # log_evidences = []
    # log_evidences_std = []
    # for idx in range(20):
    #     nde_sampler.run_sampler(num_samples=settings["num_samples"])
    #     nde_sampler.importance_sample(
    #         num_processes=settings.get("num_processes", 1),
    #         time_marginalization_kwargs=time_marginalization_kwargs,
    #         phase_marginalization_kwargs=phase_marginalization_kwargs,
    #     )
    #     log_evidences.append(nde_sampler.log_evidence)
    #     log_evidences_std.append(nde_sampler.log_evidence_std)
    # import numpy as np
    # log_evidences = np.array(log_evidences)
    # log_evidences_std = np.array(log_evidences_std)
    # print(np.std(log_evidences) / np.mean(log_evidences_std))

    if synthetic_phase:
        log.info("Sampling synthetic phase.")
        result.sample_synthetic_phase(synthetic_phase_kwargs)

    log.info("Importance sampling.")
    result.importance_sample(
        num_processes=settings.get("num_processes", 1),
        time_marginalization_kwargs=time_marginalization_kwargs,
        phase_marginalization_kwargs=phase_marginalization_kwargs,
        calibration_marginalization_kwargs=calibration_marginalization_kwargs,
    )
    result.print_summary()
    result.to_file(file_name=Path(outdir, "dingo_samples_weighted.hdf5"))

    # Diagnostics
    diagnostics_dir = join(outdir, "IS-diagnostics")
    if not exists(diagnostics_dir):
        makedirs(diagnostics_dir)
    log.info("Plotting diagnostics.")
    plot_diagnostics(
        result,
        diagnostics_dir,
        num_processes=settings.get("num_processes", 1),
        **settings.get("slice_plots", {}),
    )


if __name__ == "__main__":
    main()
