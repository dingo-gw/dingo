"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""
from pathlib import Path

import yaml
from os import rename, makedirs
from os.path import dirname, join, isfile, exists
import argparse

from dingo.core.models import PosteriorModel
from dingo.gw.result import Result
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.importance_sampling.diagnostics import plot_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Importance sampling (IS) for dingo models."
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Path to settings file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for the unconditional nde and IS results.",
    )
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = dirname(args.settings)

    return args


def main():
    # parse args, load settings, load dingo parameter samples
    args = parse_args()
    with open(args.settings, "r") as fp:
        settings = yaml.safe_load(fp)
    try:
        result = Result(file_name=settings["parameter_samples"])
    except KeyError:
        # except statement for backward compatibility
        result = Result(file_name=settings["nde"]["data"]["parameter_samples"])
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
        nde_name = settings["nde"].get(
            "path", join(args.outdir, f"nde-{event_name}.pt")
        )
        if isfile(nde_name):
            print(f"Loading nde at {nde_name} for event {event_name}.")
            nde = PosteriorModel(
                nde_name,
                device=settings["nde"]["training"]["device"],
                load_training_info=False,
            )
        else:
            print(f"Training new nde for event {event_name}.")
            nde = result.train_unconditional_flow(
                inference_parameters,
                settings["nde"],
                train_dir=args.outdir,
            )
            print(f"Renaming trained nde model to {nde_name}.")
            rename(join(args.outdir, "model_latest.pt"), nde_name)

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
        print(f"Sampling synthetic phase.")
        result.sample_synthetic_phase(synthetic_phase_kwargs)

    print(f"Importance sampling.")
    result.importance_sample(
        num_processes=settings.get("num_processes", 1),
        time_marginalization_kwargs=time_marginalization_kwargs,
        phase_marginalization_kwargs=phase_marginalization_kwargs,
        calibration_marginalization_kwargs=calibration_marginalization_kwargs,
    )
    result.print_summary()
    result.to_file(file_name=Path(args.outdir, "dingo_samples_weighted.hdf5"))

    # Diagnostics
    diagnostics_dir = join(args.outdir, "IS-diagnostics")
    if not exists(diagnostics_dir):
        makedirs(diagnostics_dir)
    print("Plotting diagnostics.")
    plot_diagnostics(
        result,
        diagnostics_dir,
        num_processes=settings.get("num_processes", 1),
        **settings.get("slice_plots", {}),
    )


if __name__ == "__main__":
    main()
