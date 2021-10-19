from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.waveform_dataset import WaveformDataset
from bilby.gw.prior import BBHPriorDict
from dingo.gw.prior_split import default_intrinsic_dict

import pytest
import numpy as np


@pytest.fixture
def setup_waveform_generator():
    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 4.0, 'window_factor': 1.0}
    domain = UniformFrequencyDomain(**domain_kwargs)
    priors = BBHPriorDict(default_intrinsic_dict)
    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, domain, reference_frequency=20.0)
    return waveform_generator, priors


def test_waveform_dataset_generation(setup_waveform_generator):
    waveform_generator, priors = setup_waveform_generator
    wd = WaveformDataset(prior=priors, waveform_generator=waveform_generator)
    n_waveforms = 10
    wd.generate_dataset(size=n_waveforms)
    assert len(wd) == n_waveforms
    assert len(wd[9]['waveform']['h_plus']) == len(waveform_generator.domain)


def test_waveform_dataset_save_load(setup_waveform_generator):
    waveform_generator, priors = setup_waveform_generator

    # Generate a dataset using the first waveform dataset object
    wd = WaveformDataset(prior=priors, waveform_generator=waveform_generator)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)
    filename = 'waveform_dataset.h5'
    wd.save(filename)

    # Now load this dataset using a second waveform dataset object
    wd2 = WaveformDataset()
    wd2.load(filename)

    # check that domain is loaded correctly
    assert wd2.domain.__dict__ == wd.domain.__dict__

    # Check that polarization data is unchanged
    idx = np.random.randint(0, n_waveforms)
    assert all([np.allclose(wd[idx]['parameters'][k], wd2[idx]['parameters'][k])
                for k in wd[idx]['parameters'].keys()])
    h0 = np.linalg.norm(wd[idx]['waveform']['h_plus'])
    assert np.allclose(wd[idx]['waveform']['h_plus'] / h0, wd2[idx]['waveform']['h_plus'] / h0)
    assert np.allclose(wd[idx]['waveform']['h_cross'] / h0, wd2[idx]['waveform']['h_cross'] / h0)
