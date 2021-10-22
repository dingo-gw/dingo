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
import pickle
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

from dingo.gw.domains import build_domain
from dingo.gw.waveform_dataset import SVDBasis
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.api import setup_logger, logger


def setup(settings_file: str) -> WaveformGenerator:
    """
    Load parameters, create domain and waveform generator.

    Parameters
    ----------
    settings_file:
        yaml file which contains options for the parameter prior.
        (Waveform domain and model settings are ignored.)
    """
    with open(settings_file, 'r') as fp:
        settings = yaml.safe_load(fp)

    domain = build_domain(settings['domain_settings'])
    waveform_generator = WaveformGenerator(settings['waveform_generator_settings']['approximant'],
                                           domain,
                                           settings['reference_frequency'])
    return waveform_generator


def read_parameter_samples(filename: str, sl: slice = None):
    """
    Read intrinsic parameter samples from a file.

    Parameters
    ----------
    filename: str
        Supported file formats are:
        '.pkl': a pickle of a dictionary of arrays
        '.npy': a structured numpy array
    sl: slice
        A slice object for selecting a subset of parameter samples
    """
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.pkl':
        with open(filename, 'rb') as fp:
            parameters = pickle.load(fp)  # dict of arrays
    elif file_extension == '.npy':
        parameters = np.load(filename)  # structured array
    else:
        raise ValueError(f'Only .pkl or .npy format supported, but got {filename}')

    return pd.DataFrame(parameters)[sl]


def generate_polarizations_task_fun(args: Tuple,
                                    waveform_generator: WaveformGenerator = None):
    """
    Picklable wrapper function for parallel waveform generation.

    Parameters
    ----------
    args:
        A tuple of (index, pandas.core.series.Series)
    waveform_generator:
        A WaveformGenerator instance
    """
    parameters = args[1].to_dict()
    return waveform_generator.generate_hplus_hcross(parameters)


def generate_dataset(waveform_generator: WaveformGenerator,
                     parameter_samples: pd.DataFrame,
                     pool: mp.Pool = None) -> pd.DataFrame:
    """Generate a waveform dataset, optionally in parallel.

    Parameters
    ----------
    waveform_generator: WaveformGenerator
        A WaveformGenerator instance
    parameter_samples: pd.DataFrame
        Intrinsic parameter samples
    pool: multiprocessing.Pool
        Optional pool of workers for parallel generation
    """
    logger.info('Generating waveform polarizations ...')
    if pool is not None:
        task_data = parameter_samples.iterrows()
        f = partial(generate_polarizations_task_fun,
                    waveform_generator=waveform_generator)
        wf_list_of_dicts = pool.map(f, task_data)
    else:
        wf_list_of_dicts = [waveform_generator.generate_hplus_hcross(p.to_dict())
                            for _, p in tqdm(parameter_samples.iterrows())]
    polarization_dict = {k: [wf[k] for wf in wf_list_of_dicts] for k in ['h_plus', 'h_cross']}
    return pd.DataFrame(polarization_dict)


def generate_waveforms(waveform_generator: WaveformGenerator,
                       parameters_file: str,
                       sl: slice = None, pool: mp.Pool = None):
    """
    Generate (a subset of) waveform polarizations from a parameter file.

    Parameters
    ----------
    waveform_generator:
        A WaveformGenerator instance.
    parameters_file:
        A parameter file saved by generate_parameters.py
    sl:
        Optionally select a slice of parameters from the full set to generate.
    pool: multiprocessing.Pool
        Optional pool of workers for parallel generation
    """
    parameter_samples = read_parameter_samples(parameters_file, sl)
    return generate_dataset(waveform_generator, parameter_samples, pool=pool)


def save_polarizations(waveform_polarizations: pd.DataFrame, idx: int,
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
        pol_arrays = {k: np.vstack(v.to_numpy().T)
                      for k, v in waveform_polarizations.items()}
        pol_dict = {k: basis.fseries_to_basis_coefficients(v)
                    for k, v in pol_arrays.items()}
        for k, v in pol_dict.items():
            np.save(f'{k}_coeff_{idx}.npy', v)
    else:
        pol_dict = {k: np.vstack(v.to_numpy().T)
                    for k, v in waveform_polarizations.items()}
        for k, v in pol_dict.items():
            np.save(f'{k}_full_{idx}.npy', v)


def parse_args():
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(description='Generate waveform polarizations.')
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing the settings file which '
                             'specifies the prior.'
                             'Write generated waveforms to this directory')
    parser.add_argument('--settings_file', type=str, default='settings.yaml')
    parser.add_argument('--parameters_file', type=str, default='parameters.npy')
    parser.add_argument('--basis_file', type=str, default='polarization_basis.npy')
    parser.add_argument('--num_wf_per_process', type=int, default=1,
                        help='Number of waveforms to generate per process.')
    parser.add_argument('--process_id', type=int, default=0,
                        help='Select slice of waveform parameter array for '
                             'which to generate waveforms.')
    parser.add_argument('--use_compression', default=False, action='store_true',
                        help='If specified, save polarizations projected basis '
                             'specified by --basis_file.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads to use in multiprocessing pool for '
                             'parallel waveform generation.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(args.waveforms_directory)
    setup_logger(outdir='.', label='generate_waveform', log_level="INFO")
    logger.info('Executing generate_waveforms:')

    waveform_generator = setup(args.settings_file)

    # Grab slice of waveform parameter array
    idx_start = args.num_wf_per_process * args.process_id
    idx_stop = idx_start + args.num_wf_per_process
    logger.info(f'Generating {args.num_wf_per_process} waveforms for '
                'chunk {args.process_id} of the parameter array.')
    logger.info(f'Slice [{idx_start}: {idx_stop}]')

    if not os.path.exists(args.parameters_file):
        raise ValueError(f'Could not find parameter file {args.parameters_file}.')
    num_samples = len(np.load(args.parameters_file))
    if (idx_start > num_samples) or (idx_stop > num_samples):
        raise ValueError(f'Specified chunk lies outside of parameter '
                         'array of length {num_samples}.')

    if args.num_threads > 1:
        # Parallel execution
        mp.freeze_support()
        logger.info(f'Running in parallel on {args.num_threads} threads.')
        with mp.Pool(processes=args.num_threads) as pool:
            waveform_polarizations = generate_waveforms(
                waveform_generator, args.parameters_file,
                slice(idx_start, idx_stop), pool=pool)
    else:
        # Serial execution
        waveform_polarizations = generate_waveforms(
            waveform_generator, args.parameters_file,
            slice(idx_start, idx_stop), pool=None)

    logger.info(f'Saving waveform polarizations to .npy file for '
                'chunk: {args.process_id}.')
    save_polarizations(waveform_polarizations, args.process_id,
                       args.use_compression, args.basis_file)
    logger.info('generate_waveforms: all done.')


if __name__ == "__main__":
    main()

