import argparse
import textwrap
from os.path import join

import yaml

from dingo.gw.noise_dataset.estimation import download_and_estimate_psds, get_time_segments
from dingo.gw.noise_dataset.generate_dataset_dag import create_dag
from dingo.gw.noise_dataset.utils import merge_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate an ASD dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path where the PSD data is to be stored. Must contain a 'settings.yaml' file.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        default=None,
        help="Optional path to a settings file in case two different datasets are generated in the sam directory",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def generate_dataset():

    args = parse_args()

    # Load settings
    settings_file = (
        args.settings_file
        if args.settings_file is not None
        else join(args.data_dir, "asd_dataset_settings.yaml")
    )
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    data_dir = args.data_dir

    time_segments = get_time_segments(data_dir, settings["dataset_settings"])

    if "condor" in settings["local"]:
        # configure_runs(time_segments_file)
        dagman = create_dag(data_dir, settings_file, time_segments)

        try:
            dagman.visualize(join(data_dir, "tmp", "condor", "ASD_dataset_generation_workflow.png"))
        except:
            pass

        dagman.build()
        print(f"DAG submission file written.")

    else:
        download_and_estimate_psds(
            args.data_dir, settings, time_segments, verbose=args.verbose
        )
        merge_datasets(data_dir)
