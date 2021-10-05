"""
Generate waveform dataset

Step 1: Generate and save parameter array

Save as parameters.pkl, parameters.npy.
"""

import argparse
import os
import pickle

import numpy as np
import yaml

from dingo.api import build_prior, structured_array_from_dict_of_arrays
from dingo.api import setup_logger, logger


def generate_parameters(settings_file: str, n_samples: int, parameters_file: str):
    """
    Parse settings file, set up priors, and draw samples from the intrinsic prior.
    Save parameters as .pkl and .npy.

    Parameters
    ----------
    settings_file:
        yaml file which contains options for the parameter prior.
        (Waveform domain and model settings are ignored.)

    n_samples:
        Number of parameter samples to generate

    parameters_file:
        The name of the parameter output file.
        Must end in either '.pkl' or '.npy'.
    """
    # Load settings
    with open(settings_file, 'r') as fp:
        settings = yaml.safe_load(fp)

    # Build prior distribution
    prior_settings = settings['prior_settings']
    prior = build_prior(prior_settings['intrinsic_parameters'],
                        prior_settings['extrinsic_parameters_reference_values'],
                        add_extrinsic_priors=True)

    # Draw prior samples
    parameter_samples_dict = prior.sample_intrinsic(size=n_samples, add_reference_values=True)

    # Save parameter file
    _, file_extension = os.path.splitext(parameters_file)
    if file_extension == '.pkl':
        logger.info('Saving parameters as pickle of dict of arrays.')
        with open(parameters_file, 'wb') as fp:
            pickle.dump(parameter_samples_dict, fp, pickle.HIGHEST_PROTOCOL)
    elif file_extension == '.npy':
        logger.info('Saving parameters as structured numpy array.')
        parameter_samples_arr = structured_array_from_dict_of_arrays(parameter_samples_dict)
        np.save(parameters_file, parameter_samples_arr)
    else:
        raise ValueError(f'Unsupported parameter file format {file_extension}.')


def main():
    parser = argparse.ArgumentParser(description='Generate waveform parameter file.')
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing the settings file which specifies the prior.')
    parser.add_argument('--settings_file', type=str, default='settings.yaml')
    parser.add_argument('--parameters_file', type=str, default='parameters.npy',
                        help='Write parameter samples to this file.')
    parser.add_argument('--n_samples', type=int, default=1)
    args = parser.parse_args()

    os.chdir(args.waveforms_directory)
    setup_logger(outdir='.', label='generate_parameters', log_level="INFO")
    logger.info('Executing generate_parameters:')

    generate_parameters(args.settings_file, args.n_samples, args.parameters_file)
    logger.info(f'Successfully generated {args.n_samples} parameters and saved to {args.parameters_file}.')


if __name__ == "__main__":
    main()

