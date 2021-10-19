from dingo.gw.domains import UniformFrequencyDomain, TimeDomain, build_domain
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

def test_FD_domain_dict(uniform_FD_params):
    p = uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    domain2 = build_domain(domain.domain_dict)
    assert domain.__dict__ == domain2.__dict__

def test_FD_truncation(uniform_FD_params):
    p = uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    N = len(domain)
    domain.initialize_truncation((40,1024))
    assert domain._truncated_sample_frequencies[0] == 40
    assert domain._truncated_sample_frequencies[-1] == 1024
    # test that array can be truncated with automatic axis selection
    a = np.random.rand(3, N, 4, 2)
    a_truncated = domain.truncate_data(a)
    assert a_truncated.shape[1] == domain._truncation_num_bins
    assert a_truncated.shape[0] == a.shape[0]
    assert a_truncated.shape[2:] == a.shape[2:]
    assert np.all(a[:,domain._truncation_idx_lower:domain._truncation_idx_upper]
                  == a_truncated)
    assert not np.all(a[:,0:domain._truncation_num_bins] == a_truncated)
    # test that axis can be selected manually
    a_truncated_2 = domain.truncate_data(a, axis=1)
    assert np.all(a_truncated == a_truncated_2)
    # test that errors are raised as intended
    with pytest.raises(ValueError):
        domain.truncate_data(a, axis=0)
    with pytest.raises(ValueError):
        domain.truncate_data(np.zeros((10, N-1, N+1)))
    with pytest.raises(ValueError):
        domain.truncate_data(np.zeros((10, N, N)))
    # test that manual axis selection works in the above case
    assert domain.truncate_data(np.zeros((10, N, N)), axis=1).shape == \
           (10, domain._truncation_num_bins, N)


def test_TD():
    time_duration, sampling_rate = 4.0, 1.0/4096.0
    domain = TimeDomain(time_duration, sampling_rate)
    delta_t = 1.0 / sampling_rate
    n = time_duration / delta_t
    times_expected = np.arange(n) * delta_t
    assert np.linalg.norm(domain() - times_expected) < 1e-15
