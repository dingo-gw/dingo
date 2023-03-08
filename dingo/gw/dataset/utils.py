import argparse
import textwrap
import copy
import pandas as pd
import numpy as np
import yaml
from typing import List

from dingo.gw.SVD import SVDBasis
from dingo.gw.dataset.generate_dataset import train_svd_basis
from dingo.gw.dataset.waveform_dataset import WaveformDataset


def merge_datasets(dataset_list: List[WaveformDataset]) -> WaveformDataset:
    """
    Merge a collection of datasets into one.

    Parameters
    ----------
    dataset_list : list[WaveformDataset]
        A list of WaveformDatasets. Each item should be a dictionary containing
        parameters and polarizations.

    Returns
    -------
    WaveformDataset containing the merged data.
    """

    print(f"Merging {len(dataset_list)} datasets into one.")

    # This ensures that all the keys are copied into the new dataset. The "extensive"
    # parts of the dataset (parameters, waveforms) will be overwritten by the combined
    # datasets, whereas the "intensive" parts (e.g., SVD basis, settings) will take the
    # values in the *first* dataset in the list.
    merged_dict = copy.deepcopy(dataset_list[0].to_dictionary())

    merged_dict["parameters"] = pd.concat([d.parameters for d in dataset_list])
    merged_dict["polarizations"] = {}
    polarizations = list(dataset_list[0].polarizations.keys())
    for pol in polarizations:
        # We pop the data array off of each of the polarizations dicts to save memory.
        # Otherwise, this operation doubles the total amount of memory used. This is
        # destructive to the original datasets.
        merged_dict["polarizations"][pol] = np.vstack(
            [d.polarizations.pop(pol) for d in dataset_list]
        )

    # Update the settings based on the total number of samples.
    merged_dict["settings"]["num_samples"] = len(merged_dict["parameters"])

    merged = WaveformDataset(dictionary=merged_dict)

    return merged


def merge_datasets_cli():
    """
    Command-line function to combine a collection of datasets into one. Used for
    parallelized waveform generation.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Combine a collection of datasets into one.
        
        Datasets must be in sequentially-labeled HDF5 files with a fixed prefix. 
        The settings for the new dataset will be based on those of the first file. 
        Optionally, replace the settings with those specified in a YAML file.
        """
        ),
    )
    parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix of sequential files names."
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        required=True,
        help="Total number of datasets to merge.",
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Name of file for new dataset."
    )
    parser.add_argument(
        "--settings_file", type=str, help="YAML file containing new dataset settings."
    )
    args = parser.parse_args()

    dataset_list = []
    for i in range(args.num_parts):
        file_name = args.prefix + str(i) + ".hdf5"
        dataset_list.append(WaveformDataset(file_name=file_name))
    merged_dataset = merge_datasets(dataset_list)

    # Optionally, update the settings file based on that provided at command line.
    if args.settings_file is not None:
        with open(args.settings_file, "r") as f:
            settings = yaml.safe_load(f)
        # Make sure num_samples is correct
        settings["num_samples"] = len(merged_dataset.parameters)
        merged_dataset.settings = settings

    merged_dataset.to_file(args.out_file)
    print(
        f"Complete. New dataset consists of {merged_dataset.settings['num_samples']} "
        f"samples."
    )


def build_svd_cli():
    """
    Command-line function to build an SVD based on an uncompressed dataset file.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Build an SVD basis based on a set of waveforms.
        """
        ),
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="HDF5 file containing training waveforms.",
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Number of basis elements to keep."
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Name of file for saving SVD."
    )
    parser.add_argument(
        "--num_train",
        type=int,
        help="Number of waveforms to use for training SVD. "
        "Remainder are used for validation.",
    )
    args = parser.parse_args()

    dataset = WaveformDataset(file_name=args.dataset_file)
    if args.num_train is None:
        n_train = len(WaveformDataset)
    else:
        n_train = args.num_train

    basis, n_train, n_test = train_svd_basis(dataset, args.size, n_train)
    # FIXME: This is not an ideal treatment. We should update the waveform generation
    #  to always provide the requested number of waveforms.
    print(
        f"SVD basis trained based on {n_train} waveforms and validated on {n_test} "
        f"waveforms. Note that if this differs from number requested, it will not be "
        f"reflected in the settings file. This is likely due to EOB failure to "
        f"generate certain waveforms."
    )
    basis.to_file(args.out_file)
