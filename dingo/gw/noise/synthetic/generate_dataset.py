import argparse
import copy
import textwrap
import numpy as np
from typing import Dict

import yaml

from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.synthetic.asd_parameterization import parameterize_asd_dataset
from dingo.gw.noise.synthetic.asd_sampling import KDE, get_rescaling_params
from dingo.gw.noise.synthetic.utils import reconstruct_psds_from_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a synthetic noise ASD dataset from an existing dataset of real ASDs. ASDs can be parameterized to generate
        smooth PSDs, e.g. to obtain a distribution over ASDs similar to BayesWave ASDs, or we can create a dataset over
        synthetic PSDs to augment the training distribution and enhance robustness. In particular, this allows us to 
        shift a distribution over ASDs from a source to a target observing run, where insufficient data is available.
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
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to draw from the parameterized ASDs",
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


def generate_dataset(real_dataset, settings: Dict, num_samples, num_processes: int, verbose: bool):
    """
    Generate a synthetic ASD dataset from an existing dataset of real ASDs.

    Parameters
    ----------
    real_dataset : ASDDataset
        Existing dataset of real ASDs.
    settings : dict
        Dictionary containing the settings for the parameterization and sampling.
    num_processes : int
        Number of processes to use in pool for parallel parameterization.
    verbose : bool
        Whether to print progress information.

    """
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

    sampling_settings = settings.get("sampling_settings")
    if sampling_settings:
        kde = KDE(parameters_dict, sampling_settings)
        kde.fit()
        rescaling_params = None
        if "rescaling_asd_paths" in sampling_settings:
            rescaling_params = get_rescaling_params(
                sampling_settings["rescaling_asd_paths"],
                settings["parameterization_settings"],
            )
        parameters_dict = kde.sample(num_samples, rescaling_params)
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

    synthetic_dataset_dict["asd_parameterizations"] = parameters_dict
    synthetic_dataset_dict["asds"] = asds_dict

    return ASDDataset(
        dictionary=synthetic_dataset_dict
    )


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    real_dataset = ASDDataset(file_name=args.asd_dataset)
    synthetic_dataset = generate_dataset(
        real_dataset, settings, args.num_samples, args.num_processes, args.verbose
    )

    synthetic_dataset.to_file(args.out_file)


if __name__ == "__main__":
    main()
