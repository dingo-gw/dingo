from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.parameters import GWPriorDict
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.waveform_dataset import WaveformDataset

import pytest
import numpy as np


@pytest.fixture
def setup_waveform_generator():
    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 4.0, 'window_factor': 1.0}
    domain = UniformFrequencyDomain(**domain_kwargs)
    priors = GWPriorDict(geocent_time_ref=1126259642.413, luminosity_distance_ref=500.0,
                         reference_frequency=20.0)
    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, domain)
    return waveform_generator, priors


def test_waveform_dataset_generation(setup_waveform_generator):
    waveform_generator, priors = setup_waveform_generator
    wd = WaveformDataset(priors=priors, waveform_generator=waveform_generator)
    n_waveforms = 10
    wd.generate_dataset(size=n_waveforms)
    assert len(wd) == n_waveforms
    assert len(wd[9]['waveform']['h_plus']) == len(waveform_generator.domain)


def test_waveform_dataset_save_load(setup_waveform_generator):
    waveform_generator, priors = setup_waveform_generator

    # Generate a dataset using the first waveform dataset object
    wd = WaveformDataset(priors=priors, waveform_generator=waveform_generator)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)
    filename = 'waveform_dataset.h5'
    wd.save(filename)

    # Now load this dataset using a second waveform dataset object
    wd2 = WaveformDataset(priors=priors, waveform_generator=waveform_generator)
    wd2.load(filename)

    # Check that polarization data is unchanged
    idx = np.random.randint(0, n_waveforms)
    assert all([np.allclose(wd[idx]['parameters'][k], wd2[idx]['parameters'][k])
                for k in wd[idx]['parameters'].keys()])
    h0 = np.linalg.norm(wd[idx]['waveform']['h_plus'])
    assert np.allclose(wd[idx]['waveform']['h_plus'] / h0, wd2[idx]['waveform']['h_plus'] / h0)
    assert np.allclose(wd[idx]['waveform']['h_cross'] / h0, wd2[idx]['waveform']['h_cross'] / h0)
