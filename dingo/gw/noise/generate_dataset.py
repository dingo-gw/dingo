import argparse
import textwrap
from os.path import join

import yaml

from dingo.gw.noise.asd_estimation import (
    download_and_estimate_psds,
    get_time_segments,
)
from dingo.gw.noise.generate_dataset_dag import create_dag
from dingo.gw.noise.utils import merge_datasets
from dingo.gw.noise.asd_sampling import KDE


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
        "--out_name",
        type=str,
        default=None,
        help="File name of resulting ASD dataset",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--override", action="store_true")

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

        dagman = create_dag(data_dir, settings_file, time_segments, args.out_name, args.override)

        try:
            dagman.visualize(
                join(data_dir, "tmp", "condor", "ASD_dataset_generation_workflow.png")
            )
        except:
            pass

        dagman.build()
        print(f"DAG submission file written.")

    else:

        download_and_estimate_psds(
            args.data_dir, settings, time_segments, verbose=args.verbose, override=args.override
        )
        dataset = merge_datasets(
            args.data_dir,
            settings["dataset_settings"],
            time_segments,
            args.out_name,
        )

        sampling_settings = settings.get("sampling_settings", None)
        if sampling_settings:
            kde = KDE(dataset, sampling_settings)
            kde.fit()
            dataset = kde.sample()

            filename = args.out_name
            if filename is None:
                run = settings["dataset_settings"]["observing_run"]
                filename = join(data_dir, f"asds_{run}.hdf5")
            dataset.to_file(filename)

