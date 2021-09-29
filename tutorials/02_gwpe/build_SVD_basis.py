"""
Generate waveform dataset

Step 3: Build SVD basis

Load hp, hc polarizations from .npy files
Save SVD basis in .npy
"""

import os
import pickle
import argparse
import glob

import numpy as np
import yaml

from tqdm import tqdm

from dingo.api import build_domain
from dingo.gw.waveform_dataset import WaveformDataset, SVDBasis
from dingo.gw.waveform_generator import WaveformGenerator


def load_polarizations_for_index(idx_start: int):
    """
    Load waveform polarizations for given index and return
    an array of stacked polarizations of shape (2*num_wfs, num_freqs).

    Here, num_wfs are the number of waveforms stored in the file and
    num_freqs the number of frequency nodes.

    Parameters
    ----------
    idx_start:
        Select data file which corresponds to a slice
        (idx_start, idx_start + num_wfs) of the full parameter array
    """
    pol_dict = {k: np.load(f'{k}_full_{idx_start}.npy') for k in ['h_plus', 'h_cross']}
    pol_stacked = np.vstack(list(pol_dict.values()))
    return pol_stacked


def find_chunk_number(parameters_file: str):
    """
    Return the number of chunks the parameter array was divided into.

    Parameters
    ----------
    parameters_file:
        .npy file containing the full parameter array
    """
    parameters = np.load(parameters_file)
    chunk_size = load_polarizations_for_index(0).shape[0] // 2
    num_chunks = len(parameters) // chunk_size

    # Sanity check
    num_files_h_plus = len(glob.glob('h_plus_full_*.npy'))
    if num_files_h_plus != num_chunks:
        raise ValueError(f'Expected {num_chunks} chunks, but found {num_files_h_plus} files "h_plus_full_*.npy"!')

    return num_chunks, chunk_size


def create_basis(num_chunks: int, outfile: str, rb_max: int = 0):
    """
    Create and save SVD basis

    Parameters
    ----------
    num_chunks:
        Number of polarization data chunks the full parameter array was split into
    outfile:
        Output file for the SVD basis V matrix
    rb_max:
        Truncate the SVD at this size
    """
    print('Load polarization data for all chunks ...')
    data = np.vstack([load_polarizations_for_index(idx) for idx in tqdm(np.arange(num_chunks))])

    print('Creating basis ...', end='')
    basis = SVDBasis()
    basis.generate_basis(data, rb_max)
    basis.to_file(outfile)
    print(' Done.')
    return basis.n, basis.V.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SVD basis from waveform polarizations.')
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing the settings file which specifies the prior.')
    parser.add_argument('--parameters_file', type=str, required=True)  # only for sanity check
    parser.add_argument('--basis_file', type=str, default='polarization_basis.npy')
    parser.add_argument('--rb_max', type=int, default=0,
                        help='Truncate the SVD basis at this size. No truncation if zero.')
    args = parser.parse_args()

    os.chdir(args.waveforms_directory)
    num_chunks, chunk_size = find_chunk_number(args.parameters_file)
    n, V_shape = create_basis(num_chunks, args.basis_file, args.rb_max)
    print(f'Created SVD basis of size {n} from {num_chunks} chunks of size {chunk_size}.')
    print(f'V matrix of shape {V_shape} saved to {args.basis_file}.')
