"""
Generate waveform dataset

Step 2 / Step 4: generate waveforms for subset of parameter array

Read parameters.npy
Only a chunk of specified size (a slice) of the full parameter array
is used to allow for parallel generation and to limit memory consumption.
Optionally read an SVD basis to compress the data.
Save polarizations in compressed or uncompressed form to .npy files.
"""

import argparse
import os

import numpy as np
import yaml

from dingo.api import build_domain
from dingo.gw.waveform_dataset import WaveformDataset, SVDBasis
from dingo.gw.waveform_generator import WaveformGenerator


def setup_waveform_dataset(settings_file: str) -> WaveformDataset:
    """
    Load parameters, create domain and waveform generator and return
    a waveform dataset object.

    Parameters
    ----------
    settings_file:
        yaml file which contains options for the parameter prior.
        (Waveform domain and model settings are ignored.)
    """
    with open(settings_file, 'r') as fp:
        settings = yaml.safe_load(fp)

    domain = build_domain(settings['domain_settings'])
    waveform_generator = WaveformGenerator(settings['waveform_generator_settings']['approximant'], domain)
    wd = WaveformDataset(priors=None, waveform_generator=waveform_generator, transform=None)
    return wd


def generate_waveforms(wd: WaveformDataset, parameters_file: str,
                       sl: slice = None):
    """
    Generate (a subset of) waveform polarizations from a parameter file.

    Parameters
    ----------
    wd:
        Waveform data set instance.
    parameters_file:
        A parameter file saved by generate_parameters.py
    sl:
        Optionally select a slice of parameters from the full set to generate.
    """
    wd.read_parameter_samples(parameters_file, sl)
    wd.generate_dataset()


def save_polarizations(wd: WaveformDataset, idx: int,
                       use_compression: bool, basis_file: str = None):
    """
    Save polarizations in .npy files.

    Parameters
    ----------
    wd:
        Waveform data set instance
    idx:
        Index of this chunk in the full parameter array
    use_compression:
        Whether to project the polarizations onto a basis
    basis_file:
        File containing an SVD basis. Only used if use_compression == True
        File names indicate whether the data are full frequency series ('full')
        or SVD projection coefficients ('coeff').
        The chunk index in the full parameter array is appended to the file name.
    """
    # TODO: Would it be better to save to HDF5 and include the parameters used?
    if use_compression:
        basis = SVDBasis()
        basis.from_file(basis_file)
        pol_dict = wd.get_compressed_polarizations(basis)
        for k, v in pol_dict.items():
            np.save(f'{k}_coeff_{idx}.npy', v)
    else:
        pol_dict = wd.get_polarizations()
        for k, v in pol_dict.items():
            np.save(f'{k}_full_{idx}.npy', v)


if __name__ == "__main__":
    # When calling this make sure to limit the number of waveforms per job
    # TODO: Should use multi-processing?
    #  Should check memory usage and save a logfile
    #  should generate batches of waveforms so that batch size fits into memory, loop over batches
    #  pass indices of current batches
    parser = argparse.ArgumentParser(description='Generate waveform polarizations.')
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing the settings file which specifies the prior.'
                             'Write generated waveforms to this directory')
    parser.add_argument('--settings_file', type=str, default='settings.yaml')
    parser.add_argument('--parameters_file', type=str, default='parameters.npy')
    parser.add_argument('--num_wf_per_process', type=int, default=1)
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--use_compression', default=False, action='store_true',
                        help='Save polarizations projected basis specified by --basis_file.')
    parser.add_argument('--basis_file', type=str, default='polarization_basis.npy')
    args = parser.parse_args()

    os.chdir(args.waveforms_directory)
    wd = setup_waveform_dataset(args.settings_file)

    # FIXME: want to batch this (as a big chunk of wfs may not fit into memory), use multiprocessing
    idx_start = args.num_wf_per_process * args.process_id
    idx_stop = idx_start + args.num_wf_per_process
    print(f'Generating {args.num_wf_per_process} waveforms for chunk {args.process_id} of the parameter array.')
    print(f'Slice [{idx_start}: {idx_stop}]')

    num_samples = len(np.load(args.parameters_file))
    if (idx_start > num_samples) or (idx_stop > num_samples):
        raise ValueError(f'Specified chunk lies outside of parameter array of length {num_samples}.')
    generate_waveforms(wd, args.parameters_file, slice(idx_start, idx_stop))

    save_polarizations(wd, args.process_id, args.use_compression, args.basis_file)
