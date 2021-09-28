"""
Generate waveform dataset

Step 1: Generate and save parameter array
"""

import os
import pickle
import argparse

import numpy as np
import yaml

from dingo.api import build_prior, build_domain, structured_array_from_dict_of_arrays
from dingo.gw.waveform_dataset import WaveformDataset
from dingo.gw.waveform_generator import WaveformGenerator


def generate_parameters(waveforms_directory: str, settings_file: str, n_samples: int):
    """
    Parse settings file, set up priors, and draw samples from the intrinsic prior.
    Save parameters as .pkl and .npy

    Parameters
    ----------
    waveforms_directory: str
        Directory containing settings file.
        The generated parameters will be saved there as well.

    settings_file:
        yaml file which contains options for the parameter prior.
        (Waveform domain and model settings are ignored.)

    n_samples:
        Number of parameter samples to generate
    """
    os.chdir(waveforms_directory)

    # Load settings
    with open(settings_file, 'r') as fp:
        settings = yaml.safe_load(fp)

    # Build prior distribution
    prior_settings = settings['prior_settings']
    prior = build_prior(prior_settings['intrinsic_parameters'],
                        prior_settings['extrinsic_parameters_reference_values'],
                        add_extrinsic_priors=True)

    # Save parameter file
    parameter_samples_dict = prior.sample_intrinsic(size=n_samples, add_reference_values=True)
    # Pickle of dict of arrays
    with open('parameters.pkl', 'wb') as fp:
        pickle.dump(parameter_samples_dict, fp, pickle.HIGHEST_PROTOCOL)
    # Structured numpy array
    parameter_samples_arr = structured_array_from_dict_of_arrays(parameter_samples_dict)
    np.save('parameters.npy', parameter_samples_arr)


def test_load_parameters_and_generate_dataset(waveforms_directory: str, settings_file: str):
    """
    Function to test that parameters can be correctly loaded and used in a WaveformDataset
    """
    #os.chdir(waveforms_directory)

    # Load settings
    with open(settings_file, 'r') as fp:
        settings = yaml.safe_load(fp)

    domain = build_domain(settings['domain_settings'])
    waveform_generator = WaveformGenerator(settings['waveform_generator_settings']['approximant'], domain)
    wd = WaveformDataset(priors=None, waveform_generator=waveform_generator, transform=None)
    wd.read_parameter_samples('parameters.npy')
    wd.generate_dataset(size=42)
    print(wd[0]['waveform'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate waveform parameter file.')
    parser.add_argument('--waveforms_directory', type=str, required=True,
                        help='Directory containing the settings file which specifies the prior.')
    parser.add_argument('--settings_file', type=str, default='settings.yaml')
    parser.add_argument('--n_samples', type=int, default=1)
    args = parser.parse_args()

    generate_parameters(args.waveforms_directory, args.settings_file, args.n_samples)
    #test_load_parameters_and_generate_dataset(args.waveforms_directory, args.settings_file)
