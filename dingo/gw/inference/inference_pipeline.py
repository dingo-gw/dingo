from typing import Optional

import torch
import numpy as np
import argparse
import os
from os.path import dirname, join
from pathlib import Path
import yaml

from dingo.core.models import PosteriorModel
from dingo.core.utils.plotting import plot_corner_multi
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSampler, GWSamplerGNPE
from dingo.gw.data.data_preparation import get_event_data_and_domain, \
    parse_settings_for_raw_data
from dingo.gw.inference.visualization import load_ref_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer the posterior of a GW event using a trained dingo model.",
    )
    parser.add_argument(
        "--out_directory",
        type=str,
        default=None,
        help="Directory for output of analysis (samples, plots).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained dingo model.",
    )
    parser.add_argument(
        "--model_init",
        type=str,
        default=None,
        help="Path to trained dingo model for initialization of gnpe. Only required if "
        "gnpe is used.",
    )
    parser.add_argument(
        # TODO: this should be renamed --events or similar
        "--gps_time_event",
        type=str,
        required=True,
        nargs="+",
        help="List of GPS times of the events to be analyzed. Used to download the GW "
        "event data, or search for it in the dataset file. Can also be a string with "
        "the path to a file containing the event data.",
    )
    parser.add_argument(
        "--event_dataset",
        type=str,
        default=None,
        help="Path to dataset for GW event data. If set, GW event data is cached in "
        "this dataset, such that it only needs to be downloaded once.",
    )
    parser.add_argument(
        "--event_dataset_init",
        type=str,
        default=None,
        help="Path to dataset for GW event data, for gnpe initialization model "
        "model_init. If set, GW event data is cached in this dataset, such that it "
        "only needs to be downloaded once.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50_000,
        help="Number of posterior samples per event.",
    )
    parser.add_argument(
        "--num_gnpe_iterations",
        type=int,
        default=30,
        help="Number of gnpe iterations.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="File with path to reference samples (e.g., bilby, LALInference). If set, "
        "the corresponding samples are loaded and compared in a cornerplot.",
    )
    # settings for raw GW data
    parser.add_argument(
        "--time_psd",
        type=float,
        default=1024,
        help="Time in seconds used for PSD estimation.",
    )
    parser.add_argument(
        "--time_buffer",
        type=float,
        default=2.0,
        help="Buffer time in seconds. The analyzed strain segment extends up to "
        "gps_time_event + time_buffer.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for sample name.",
    )
    parser.add_argument(
        "--get_log_prob",
        action="store_true",
        help="If set, log_probs of samples are saved. For GNPE models, this invokes "
        "training of an unconditional density estimator to recover the density.",
    )
    parser.add_argument(
        "--save_low_latency",
        action="store_true",
        help="Only relevant for GNPE models and if args.get_log_prob is set. In that "
        "case, and if save_low_latency is set, intermediate results without log_prob "
        "are saved.",
    )
    parser.add_argument(
        "--density_settings",
        type=str,
        default=None,
        help="Path to yaml file with settings for density estimation. Only used if "
        "log_prob is requested for a gnpe model.",
    )
    parser.add_argument(
        "--exit_command",
        type=str,
        default=None,
        nargs="+",
        help="If set, run os.system(args.exit_command) before exiting.",
    )

    args = parser.parse_args()

    return args


def get_event_data(event, args, model, ref=None):
    # if event is a float: interpret it as gps time, download the data
    # else: interpret it as path to file with event data
    try:
        time_event = float(event)
    except ValueError:
        time_event = None

    if time_event is not None:
        # get raw event data, and prepare it for the network domain
        event_data, _ = get_event_data_and_domain(
            model.metadata,
            time_event,
            args.time_psd,
            args.time_buffer,
            args.event_dataset,
        )

        event_metadata = parse_settings_for_raw_data(model.metadata, args.time_psd,
                                                     args.time_buffer)

        # Put the metadata in the same format as provided by dingo_pipe data_generation.
        # (This is a bit ad hoc, should be improved.)
        event_metadata["time_event"] = time_event
        event_metadata["psd_duration"] = event_metadata.pop("time_psd")
        window = event_metadata.pop("window")
        event_metadata["T"] = window["T"]
        del event_metadata["time_segment"]
        event_metadata["window_type"] = window["type"]
        event_metadata["roll_off"] = window["roll_off"]
        event_metadata["f_min"] = model.metadata["dataset_settings"]["domain"]["f_min"]
        event_metadata["f_max"] = model.metadata["dataset_settings"]["domain"]["f_max"]

        if ref is None or time_event not in ref:
            label = f"gps-{time_event}{args.suffix}"
        else:
            name_event = ref[time_event]["event_name"]
            label = name_event + args.suffix

    else:
        # load file with event data
        event_dataset = EventDataset(file_name=event)
        event_data = event_dataset.data
        event_metadata = event_dataset.settings
        label = Path(event).stem + args.suffix

    return event_data, event_metadata, label


def prepare_log_prob(
    sampler,
    num_samples: int,
    nde_settings: dict,
    batch_size: Optional[int] = None,
    threshold_std: Optional[float] = np.inf,
    remove_init_outliers: Optional[float] = 0.0,
    low_latency_label: str = None,
    outdir: str = None,
):
    """
    Prepare gnpe sampling with log_prob. This is required, since in its vanilla
    form gnpe does not provide the density for its samples.

    Specifically, we train an unconditional neural density estimator (nde) for the
    gnpe proxies. This requires running the gnpe sampler till convergence, and
    extracting the gnpe proxies after the final gnpe iteration. The nde is trained
    to match the distribution over gnpe proxies, which provides a way of rapidly
    sampling (converged!) gnpe proxies *and* evaluating the log_prob.

    After this preparation step, self.run_sampler can leverage
    self.gnpe_proxy_sampler (which is based on the aforementioned trained nde) to
    sample gnpe proxies, such that one gnpe iteration is sufficient. The
    log_prob of
    the samples in the *joint* space (inference parameters + gnpe proxies) is then
    simply given by the sum of the corresponding log_probs (from self.model and
    self.gnpe_proxy_sampler.model).

    Parameters
    ----------
    num_samples: int
        number of samples for training of nde
    batch_size: int = None
        batch size for sampler
    threshold_std: float = np.inf
        gnpe proxies deviating by more then threshold_std standard deviations from
        the proxy mean (along any axis) are discarded.
    low_latency_label: str = None
        File label for low latency samples (= samples used for training nde).
        If None, these samples are not saved.
    outdir: str = None
        Directory in which low latency samples are saved. Needs to be set if
        low_latency_label is not None.
    """
    sampler.remove_init_outliers = remove_init_outliers
    sampler.run_sampler(num_samples, batch_size)
    if low_latency_label is not None:
        sampler.to_hdf5(label=low_latency_label, outdir=outdir)
    result = sampler.to_result()
    nde_settings["training"]["device"] = str(sampler.model.device)
    unconditional_model = result.train_unconditional_flow(
        sampler.gnpe_proxy_parameters,
        nde_settings,
        threshold_std=threshold_std,
    )

    # Prepare sampler with unconditional model as initialization. This should only use
    # one iteration and also not remove any outliers.
    sampler.init_sampler = GWSampler(model=unconditional_model)
    sampler.num_iterations = 1
    sampler.remove_init_outliers = 0.0  # Turn off for final sampler.


def analyze_event():
    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = PosteriorModel(args.model, device=device, load_training_info=False)
    epoch = model.epoch
    wf_model = model.metadata["dataset_settings"]["waveform_generator"]["approximant"]

    # create analysis directory
    if args.out_directory is None:
        args.out_directory = join(dirname(args.model), f"analysis_epoch-{epoch}")
    os.makedirs(args.out_directory, exist_ok=True)

    #
    if args.reference_file is not None:
        with open(args.reference_file, "r") as f:
            ref = yaml.safe_load(f)[wf_model]
    else:
        ref = None

    if args.model_init is not None:
        gnpe = True
        init_model = PosteriorModel(
            args.model_init, device=device, load_training_info=False
        )
        init_sampler = GWSampler(model=init_model)
        sampler = GWSamplerGNPE(
            model=model,
            init_sampler=init_sampler,
            num_iterations=args.num_gnpe_iterations,
        )
    else:
        gnpe = False
        sampler = GWSampler(model=model)

    # sample posterior for events
    for time_event in args.gps_time_event:
        print(f"Analyzing event at {time_event}.")

        event_data, event_metadata, label = get_event_data(time_event, args, model, ref)

        sampler.context = event_data
        sampler.event_metadata = event_metadata

        if gnpe and args.get_log_prob:
            # GNPE generally does not provide straightforward access to the log_prob.
            # If requested, need to train an initialization model for the GNPE proxies.
            with open(args.density_settings) as f:
                density_settings = yaml.safe_load(f)
            if args.save_low_latency:
                low_latency_label = label + "_low-latency"
            else:
                low_latency_label = None
            prepare_log_prob(
                sampler,
                batch_size=args.batch_size,
                low_latency_label=low_latency_label,
                outdir=args.out_directory,
                **density_settings,
            )
            if low_latency_label is not None and sampler.iteration_tracker.store_data:
                np.save(
                    join(
                        args.out_directory,
                        f"gnpe_trajectories_{low_latency_label}.npy",
                    ),
                    sampler.iteration_tracker.data,
                )

        sampler.run_sampler(
            args.num_samples,
            batch_size=args.batch_size,
        )

        # if no reference samples are available, simply save the dingo samples
        if ref is None or time_event not in ref:
            sampler.to_hdf5(label=label, outdir=args.out_directory)

        # if reference samples are available, save dingo samples and additionally
        # compare to the reference method in a corner plot
        else:
            ref_samples_file = ref[time_event]["reference_samples"]["file"]
            ref_method = ref[time_event]["reference_samples"]["method"]
            sampler.to_hdf5(label=label, outdir=args.out_directory)
            ref_samples = load_ref_samples(ref_samples_file)
            plot_corner_multi(
                [ref_samples, sampler.samples],
                labels=[ref_method, "Dingo"],
                filename=join(args.out_directory, f"cornerplot_{label}.pdf")
            )
    if args.exit_command:
        os.system(" ".join(args.exit_command))


if __name__ == "__main__":
    analyze_event()
