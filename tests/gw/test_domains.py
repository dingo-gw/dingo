from dingo.gw.domains import UniformFrequencyDomain, TimeDomain
import pytest
import numpy as np


@pytest.fixture
def uniform_FD_params():
    f_min = 20.0
    f_max = 4096.0
    delta_f = 1.0 / 4.0
    window_factor = 1.0
    return {'f_min': f_min, 'f_max': f_max, 'delta_f': delta_f, 'window_factor': window_factor}

def test_uniform_FD(uniform_FD_params):
    p = uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    n = int(p['f_max'] / p['delta_f']) + 1
    frequencies_expected = np.linspace(0, p['f_max'], n)
    frequencies = domain()
    assert np.linalg.norm(frequencies - frequencies_expected) < 1e-15

def test_uniform_FD_mask(uniform_FD_params):
    p = uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    mask = domain.frequency_mask
    n_masked = int((p['f_max'] - p['f_min']) / p['delta_f']) + 1
    frequencies_expected_masked = np.linspace(p['f_min'], p['f_max'], n_masked)
    frequencies_masked = domain()[mask]
    assert np.linalg.norm(frequencies_masked - frequencies_expected_masked) < 1e-15

def test_uniform_FD_noise_std(uniform_FD_params):
    p = uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    expected = np.sqrt(p['window_factor']) / np.sqrt(4.0 * p['delta_f'])
    assert np.abs(domain.noise_std - expected) < 1e-15


def test_TD():
    time_duration, sampling_rate = 4.0, 1.0/4096.0
    domain = TimeDomain(time_duration, sampling_rate)
    delta_t = 1.0 / sampling_rate
    n = time_duration / delta_t
    times_expected = np.arange(n) * delta_t
    assert np.linalg.norm(domain() - times_expected) < 1e-15
