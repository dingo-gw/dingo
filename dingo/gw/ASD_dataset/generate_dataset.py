import yaml
from os.path import join
import argparse
import textwrap
from typing import List

from dingo.gw.ASD_dataset.dataset_utils import (
    download_and_estimate_PSDs,
    create_dataset_from_files,
)


def generate_dataset():

    args = parse_args()

    # Load settings
    if args.settings is not None:
        with open(args.settings, "r") as f:
            settings = yaml.safe_load(f)
    else:
        with open(join(args.data_dir, "settings.yaml"), "r") as f:
            settings = yaml.safe_load(f)

    download_and_estimate_PSDs(
        args.data_dir, settings["dataset_settings"], verbose=args.verbose
    )

    create_dataset_from_files(args.data_dir, settings["dataset_settings"])


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
        "--settings",
        type=str,
        default=None,
        help="Optional path to a settings file in case two different datasets are generated in the sam directory",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()
