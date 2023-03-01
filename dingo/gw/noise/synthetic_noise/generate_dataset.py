import argparse
import copy
import textwrap
import numpy as np
from typing import Dict

import yaml

from dingo.gw.dataset import DingoDataset
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.synthetic_noise.asd_parameterization import parameterize_asd_dataset
from dingo.gw.noise.synthetic_noise.asd_sampling import KDE, get_rescaling_params
from dingo.gw.noise.synthetic_noise.utils import reconstruct_psds_from_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a synthetic noise ASD dataset from an existing dataset of real ASDs.
        """
        ),
    )
    parser.add_argument(
        "--asd_dataset",
        type=str,
        required=True,
        help="Path to existing ASD dataset to be parameterized and re-sampled",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel parameterization",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="synthetic_asd_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def generate_dataset(real_dataset, settings: Dict, num_processes: int, verbose: bool):
    parameters_dict = parameterize_asd_dataset(
        real_dataset,
        settings["parameterization_settings"],
        num_processes,
        verbose=verbose,
    )

    synthetic_dataset_dict = copy.deepcopy(real_dataset.to_dictionary())
    synthetic_dataset_dict["settings"]["parameterization_settings"] = settings[
        "parameterization_settings"
    ]

    sampling_settings = settings.get("sampling_settings", None)
    if sampling_settings:
        kde = KDE(parameters_dict, sampling_settings)
        kde.fit()
        rescaling_params = None
        if "rescaling_psd_paths" in sampling_settings:
            rescaling_params = get_rescaling_params(
                sampling_settings["rescaling_psd_paths"],
                settings["parameterization_settings"],
            )
        parameters_dict = kde.sample(rescaling_params)
        synthetic_dataset_dict["settings"]["sampling_settings"] = settings["sampling_settings"]

    asds_dict = {}
    for det, params in parameters_dict.items():
        psds = reconstruct_psds_from_parameters(
            params,
            real_dataset.domain,
            settings["parameterization_settings"],
        )

        asds_dict[det] = np.sqrt(
            psds[:, real_dataset.domain.min_idx : real_dataset.domain.max_idx + 1]
        )

    synthetic_dataset_dict["parameters"] = parameters_dict
    synthetic_dataset_dict["asds"] = asds_dict
    # TODO: should I make the "data_keys" an optional argument to the ASDDataset class? Such that the parameters
    # can also be stored in this way
    return DingoDataset(
        dictionary=synthetic_dataset_dict, data_keys=["asds", "gps_times", "parameters"]
    )


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    real_dataset = ASDDataset(file_name=args.asd_dataset)
    synthetic_dataset = generate_dataset(
        real_dataset, settings, args.num_processes, args.verbose
    )

    synthetic_dataset.to_file(args.out_file)


if __name__ == "__main__":
    main()
