"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""
import yaml
from os import rename, makedirs
from os.path import dirname, join, isfile, exists
from types import SimpleNamespace
import argparse

from dingo.core.models import PosteriorModel
from dingo.core.samples_dataset import SamplesDataset
from dingo.gw.inference.gw_samplers import GWSamplerUnconditional
from dingo.core.density import train_unconditional_density_estimator
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
        samples_dataset = SamplesDataset(file_name=settings["parameter_samples"])
    except KeyError:
        # except statement for backward compatibility
        samples_dataset = SamplesDataset(
            file_name=settings["nde"]["data"]["parameter_samples"]
        )
    metadata = samples_dataset.settings
    samples = samples_dataset.samples
    # for time marginalization, we drop geocent time from the samples
    inference_parameters = metadata["train_settings"]["data"][
        "inference_parameters"
    ].copy()
    time_marginalization_kwargs = settings.get("time_marginalization", None)
    time_marginalization = time_marginalization_kwargs is not None
    phase_marginalization_kwargs = settings.get("phase_marginalization", None)
    phase_marginalization = phase_marginalization_kwargs is not None
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
        event_name = str(
            metadata["event"]["time_event"]
        )  # use gps time as name for now
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
            nde = train_unconditional_density_estimator(
                samples_dataset, settings["nde"], args.outdir
            )
            print(f"Renaming trained nde model to {nde_name}.")
            rename(join(args.outdir, "model_latest.pt"), nde_name)
            nde_sampler = GWSamplerUnconditional(
                model=nde, synthetic_phase_kwargs=synthetic_phase_kwargs
            )

    else:
        nde_sampler = GWSamplerUnconditional(
            samples_dataset=samples_dataset,
            synthetic_phase_kwargs=synthetic_phase_kwargs,
        )

    # Step 2: Sample from proposal.

    print(f'Generating {settings["num_samples"]} samples from proposal distribution.')
    nde_sampler.run_sampler(num_samples=settings["num_samples"])

    # Step 3: Importance sample.
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

    print(f"Importance sampling.")
    nde_sampler.importance_sample(
        num_processes=settings.get("num_processes", 1),
        time_marginalization_kwargs=time_marginalization_kwargs,
        phase_marginalization_kwargs=phase_marginalization_kwargs,
    )
    nde_sampler.print_summary()
    nde_sampler.to_hdf5(label="weighted", outdir=args.outdir)
    samples = nde_sampler.samples

    # Diagnostics
    diagnostics_dir = join(args.outdir, "IS-diagnostics")
    if not exists(diagnostics_dir):
        makedirs(diagnostics_dir)
    print("Plotting diagnostics.")
    plot_diagnostics(
        nde_sampler,
        diagnostics_dir,
        num_processes=settings.get("num_processes", 1),
        **settings.get("slice_plots", {}),
    )


if __name__ == "__main__":
    main()
