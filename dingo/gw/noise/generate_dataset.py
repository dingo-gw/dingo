import argparse
import textwrap
from os.path import join

import yaml
import pickle

from dingo.gw.noise.asd_estimation import (
    download_and_estimate_psds,
)
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.generate_dataset_dag import create_dag
from dingo.gw.noise.utils import merge_datasets, get_time_segments


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
    parser.add_argument(
        "--time_segments_file",
        type=str,
        default=None,
        help="Optional file containing a dictionary of a list of time segments that should be used for estimating PSDs."
             "This has to be a pickle file.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="File name of resulting ASD dataset",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def generate_dataset():
    """
    Creates and saves an ASD dataset
    """
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

    if args.time_segments_file:
        with open(args.time_segments_file, "rb") as f:
            time_segments = pickle.load(f)
    else:
        time_segments = get_time_segments(data_dir, settings["dataset_settings"])

    if "condor" in settings["local"]:

        dagman = create_dag(data_dir, settings_file, time_segments, args.out_name)

        try:
            dagman.visualize(
                join(data_dir, "tmp", "condor", "ASD_dataset_generation_workflow.png")
            )
        except:
            pass

        dagman.build()
        print(f"DAG submission file written.")

    else:

        asd_filename_list = download_and_estimate_psds(
            args.data_dir, settings, time_segments, verbose=args.verbose
        )
        asd_dataset_list = {
            det: [ASDDataset(asd_file) for asd_file in asd_file_list]
            for det, asd_file_list in asd_filename_list.items()
        }
        dataset = merge_datasets(asd_dataset_list)
        filename = args.out_name
        if filename is None:
            run = settings["dataset_settings"]["observing_run"]
            filename = join(data_dir, f"asds_{run}.hdf5")
        dataset.to_file(filename)
