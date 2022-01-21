import ast
import argparse
import copy
from typing import Dict
import pandas as pd
import numpy as np
import h5py
import yaml

from dingo.core.utils.dataset_utils import recursive_hdf5_load, save_dataset
from dingo.gw.SVD import SVDBasis


def structured_array_from_dict_of_arrays(d: Dict[str, np.ndarray], fmt: str = 'f8'):
    """
    Given a dictionary of 1-D numpy arrays, create a numpy structured array
    with field names equal to the dict keys and using the specified format.

    Parameters
    ----------
    d : Dict
        A dictionary of 1-dimensional numpy arrays
    fmt : str
        Format string for the array datatype.
    """
    arr_raw = np.array(list(d.values()))
    par_names = list(d.keys())
    dtype = np.dtype([x for x in zip(par_names, [fmt for _ in par_names])])
    arr_struct = np.array([tuple(x) for x in arr_raw.T], dtype=dtype)

    # Check: TODO: move to a unit test
    arr_rec = pd.DataFrame(d).to_records()
    assert np.all([np.allclose(arr_struct[k], arr_rec[k]) for k in par_names])

    return arr_struct


def dataframe_to_structured_array(df: pd.DataFrame):
    """
    Convert a pandas DataFrame of parameters to a structured numpy array.

    Parameters
    ----------
    df:
        A pandas DataFrame
    """
    d = {k: np.array(list(v.values())) for k, v in df.to_dict().items()}
    return structured_array_from_dict_of_arrays(d)


def get_params_dict_from_array(params_array, params_inds, f_ref=None):
    """
    Transforms an array with shape (num_parameters) to a dict. This is
    necessary for the waveform generator interface.

    Parameters
    ----------
    params_array:
        Array with parameters
    params_inds:
        Indices for the individual parameter keys.
    f_ref:
        Reference frequency of approximant
    params:
        Dictionary with the parameters
    """
    if len(params_array.shape) > 1:
        raise ValueError('This function only transforms a single set of '
                         'parameters to a dict at a time.')
    params = {}
    for k, v in params_inds.items():
        params[k] = params_array[v]
    if f_ref is not None:
        params['f_ref'] = f_ref
    return params


def merge_datasets(dataset_list):

    # This ensures that all of the keys are copied into the new dataset. The
    # "extensive" parts of the dataset (parameters, waveforms) will be overwritten by
    # the combined datasets, whereas the "intensive" parts (e.g., SVD basis) will take
    # the values in the *first* dataset in the list.
    merged = copy.deepcopy(dataset_list[0])

    merged['parameters'] = np.vstack([d['parameters'] for d in dataset_list])
    merged['polarizations'] = {}
    for pol in dataset_list[0]['polarizations']:
        merged['polarizations']['pol'] = np.vstack([d['polarizations']['pol'] for d in
                                                    dataset_list])

    return merged


def merge_datasets_cli():

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--num_parts', type=int, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--settings_file', type=str)
    args = parser.parse_args()

    dataset_list = []
    for i in range(args.num_parts):
        file_name = args.prefix + str(i) + '.hdf5'
        with h5py.File(file_name, 'r') as f:
            dataset_list.append(recursive_hdf5_load(f))
    merged_dataset = merge_datasets(dataset_list)

    if args.settings_file is not None:
        with open(args.settings_file, 'r') as f:
            settings = yaml.safe_load(f)
    else:
        # If not included as an argument, just take the settings from the first dataset
        # in the merge list.
        file_name = args.prefix + '0.hdf5'
        with h5py.File(file_name, 'r') as f:
            settings = ast.literal_eval(f.attrs['settings'])

    # Update settings/num_samples to be consistent with the dataset.
    settings['num_samples'] = len(merged_dataset('parameters'))
    save_dataset(merged_dataset, settings, args.out_file)


def build_svd_cli():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    args = parser.parse_args()

    # We build the SVD based on all of the polarizations.
    polarizations = []
    with h5py.File(args.dataset_file, 'r') as f:
        for pol, data in f['polarizations'].items():
            polarizations.append(data[...])
    train_data = np.vstack(polarizations)

    basis = SVDBasis()
    basis.generate_basis(train_data, args.size)
    basis.to_file(args.out_file)

