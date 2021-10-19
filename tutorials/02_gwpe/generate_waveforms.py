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
from multiprocessing import Pool, freeze_support

from dingo.gw.domains import build_domain
from dingo.gw.waveform_dataset import WaveformDataset, SVDBasis
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.api import setup_logger, logger


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
    wd = WaveformDataset(prior=None, waveform_generator=waveform_generator, transform=None)
    return wd


def generate_waveforms(wd: WaveformDataset, parameters_file: str,
                       sl: slice = None, pool: Pool = None):
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
    wd.generate_dataset(pool=pool)


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


def main():
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
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to use in pool for parallel waveform generation')
    args = parser.parse_args()

    os.chdir(args.waveforms_directory)
    setup_logger(outdir='.', label='generate_waveform', log_level="INFO")
    logger.info('Executing generate_waveforms:')

    wd = setup_waveform_dataset(args.settings_file)

    idx_start = args.num_wf_per_process * args.process_id
    idx_stop = idx_start + args.num_wf_per_process
    logger.info(f'Generating {args.num_wf_per_process} waveforms for chunk {args.process_id} of the parameter array.')
    logger.info(f'Slice [{idx_start}: {idx_stop}]')

    num_samples = len(np.load(args.parameters_file))
    if (idx_start > num_samples) or (idx_stop > num_samples):
        raise ValueError(f'Specified chunk lies outside of parameter array of length {num_samples}.')

    if args.num_threads > 1:
        # Parallel execution
        freeze_support()
        print(f'Running in parallel on {args.num_threads} threads.')
        with Pool(processes=args.num_threads) as pool:
            generate_waveforms(wd, args.parameters_file, slice(idx_start, idx_stop), pool=pool)
    else:
        # Serial execution
        generate_waveforms(wd, args.parameters_file, slice(idx_start, idx_stop), pool=None)

    save_polarizations(wd, args.process_id, args.use_compression, args.basis_file)


if __name__ == "__main__":
    main()

