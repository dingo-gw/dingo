import torch
import argparse
import os
from os.path import dirname, join
import yaml

from dingo.core.models import PosteriorModel
from dingo.gw.inference.gw_samplers import GWSampler, GWSamplerGNPE
from dingo.gw.inference.data_preparation import get_event_data_and_domain
from dingo.gw.inference.visualization import load_ref_samples, generate_cornerplot


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
        "--gps_time_event",
        type=float,
        required=True,
        nargs="+",
        help="List of GPS times of the events to be analyzed. Used to download the GW "
        "event data, or search for it in the dataset file.",
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
        help="If set, log_probs of samples are saved. Will not work for GNPE.",
    )

    args = parser.parse_args()

    return args


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
        sampler = GWSampler(model=model)

    # sample posterior for events
    for time_event in args.gps_time_event:
        print(f"Analyzing event at gps time {time_event}.")

        # get raw event data, and prepare it for the network domain
        event_data, _ = get_event_data_and_domain(
            model.metadata,
            time_event,
            args.time_psd,
            args.time_buffer,
            args.event_dataset,
        )
        sampler.context = event_data
        sampler.event_metadata = {
            "time_event": time_event,
            "time_psd": args.time_psd,
            "time_buffer": args.time_buffer,
        }
        sampler.run_sampler(
            args.num_samples,
            batch_size=args.batch_size,
        )

        # if no reference samples are available, simply save the dingo samples
        if ref is None or time_event not in ref:
            label = f"gps-{time_event}{args.suffix}"
            sampler.to_hdf5(label=label, outdir=args.out_directory)

        # if reference samples are available, save dingo samples and additionally
        # compare to the reference method in a corner plot
        else:
            name_event = ref[time_event]["event_name"]
            ref_samples_file = ref[time_event]["reference_samples"]["file"]
            ref_method = ref[time_event]["reference_samples"]["method"]

            sampler.to_hdf5(label=name_event, outdir=args.out_directory)

            ref_samples = load_ref_samples(ref_samples_file)

            generate_cornerplot(
                {"name": ref_method, "samples": ref_samples, "color": "blue"},
                {"name": "dingo", "samples": sampler.samples, "color": "orange"},
                filename=join(
                    args.out_directory, f"cornerplot_{name_event}{args.suffix}.pdf"
                ),
            )


if __name__ == "__main__":
    analyze_event()
