import ast
import argparse
import textwrap
import copy
from typing import Dict
import pandas as pd
import numpy as np
import h5py
import yaml

from dingo.core.utils.dataset_utils import save_dataset
from dingo.core.dataset import recursive_hdf5_load
from dingo.gw.SVD import SVDBasis


def merge_datasets(dataset_list: list):
    """
    Merge a collection of datasets into one.

    Parameters
    ----------
    dataset_list : list
        A list of datasets. Each item should be a dictionary containing parameters and
        polarizations.

    Returns
    -------
    A new dataset, which is a dictionary containing parameters and polarizations.
    """

    print(f"Merging {len(dataset_list)} datasets into one.")

    # This ensures that all of the keys are copied into the new dataset. The
    # "extensive" parts of the dataset (parameters, waveforms) will be overwritten by
    # the combined datasets, whereas the "intensive" parts (e.g., SVD basis) will take
    # the values in the *first* dataset in the list.
    merged = copy.deepcopy(dataset_list[0])

    merged["parameters"] = pd.concat([d["parameters"] for d in dataset_list])
    merged["polarizations"] = {}
    for pol in dataset_list[0]["polarizations"]:
        merged["polarizations"][pol] = np.vstack(
            [d["polarizations"][pol] for d in dataset_list]
        )

    return merged


def merge_datasets_cli():
    """
    Command-line function to combine a collection of datasets into one. Used for
    parallelized waveform generation.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Combine a collection of datasets into one.
        
        Datasets must be in sequentially-labeled HDF5 files with a fixed prefix. 
        The settings for the new dataset will be based on those of the first file. 
        Optionally, replace the settings with those specified in a YAML file.
        """))
    parser.add_argument("--prefix", type=str, required=True,
                        help='Prefix of sequential files names.')
    parser.add_argument("--num_parts", type=int, required=True,
                        help='Total number of datasets to merge.')
    parser.add_argument("--out_file", type=str, required=True,
                        help='Name of file for new dataset.')
    parser.add_argument("--settings_file", type=str,
                        help='YAML file containing new dataset settings.')
    args = parser.parse_args()

    dataset_list = []
    for i in range(args.num_parts):
        file_name = args.prefix + str(i) + ".hdf5"
        with h5py.File(file_name, "r") as f:
            dataset_list.append(recursive_hdf5_load(f))
    merged_dataset = merge_datasets(dataset_list)

    if args.settings_file is not None:
        with open(args.settings_file, "r") as f:
            settings = yaml.safe_load(f)
    else:
        # If not included as an argument, just take the settings from the first dataset
        # in the merge list.
        file_name = args.prefix + "0.hdf5"
        with h5py.File(file_name, "r") as f:
            settings = ast.literal_eval(f.attrs["settings"])

    # Update settings/num_samples to be consistent with the dataset.
    settings["num_samples"] = len(merged_dataset["parameters"])
    save_dataset(merged_dataset, settings, args.out_file)

    print(f"Complete. New dataset consists of {settings['num_samples']} samples.")


def build_svd_cli():
    """
    Command-line function to build an SVD based on a file containing a collection of
    waveform polarizations.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Build an SVD basis based on a set of waveforms.
        """))
    parser.add_argument("--dataset_file", type=str, required=True,
                        help='HDF5 file containing training waveforms.')
    parser.add_argument("--size", type=int, required=True,
                        help='Number of basis elements to keep.')
    parser.add_argument("--out_file", type=str, required=True,
                        help='Name of file for saving SVD.')
    args = parser.parse_args()

    # We build the SVD based on all of the polarizations.
    print("Loading saved waveforms.")
    polarizations = []
    with h5py.File(args.dataset_file, "r") as f:
        for pol, data in f["polarizations"].items():
            polarizations.append(data[...])
    train_data = np.vstack(polarizations)

    print("Building SVD basis.")
    basis = SVDBasis()
    basis.generate_basis(train_data, args.size)
    basis.to_file(args.out_file)
