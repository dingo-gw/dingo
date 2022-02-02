import os

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
import numpy as np
import yaml
from os.path import join
import logging
import argparse
import textwrap
from typing import Dict, List
from functools import partial
from tqdm import tqdm

from dataset_utils import download_and_estimate_PSDs, create_dataset_from_files


def generate_dataset(data_dir, settings, run: str, ifos: List[str], verbose=False):

    for ifo in ifos:
        print(f"Downloading PSD data for observing run {run} and detector {ifo}")
        download_and_estimate_PSDs(data_dir, run, ifo, settings['dataset_settings'], verbose=verbose)

    create_dataset_from_files(data_dir, run, ifos, settings["dataset_settings"])


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
        "--observing_run",
        type=str,
        required=True,
        help="Observing run for which to generate the dataset",
    )
    parser.add_argument(
        "--detectors",
        type=list,
        default=["H1", "L1"],
        help="Detectors for which to generate the dataset",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel parameterisation",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Visualize progress with bars",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Load settings
    with open(join(args.data_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    generate_dataset(
        args.data_dir,
        settings,
        args.observing_run,
        args.detectors,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
