import torch

from dingo.gw.domains import FrequencyDomain, TimeDomain, build_domain
from dingo.gw.gwutils import get_window_factor
import pytest
import numpy as np


@pytest.fixture
def uniform_FD_params():
    f_min = 20.0
    f_max = 4096.0
    delta_f = 1.0 / 4.0
    return {"f_min": f_min, "f_max": f_max, "delta_f": delta_f}


@pytest.fixture
def window_setup():
    type = "tukey"
    f_s = 4096
    T = 8.0
    roll_off = 0.4
    window_kwargs = {
        "type": type,
        "f_s": f_s,
        "T": T,
        "roll_off": roll_off,
    }
    window_factor = get_window_factor(window_kwargs)
    return window_kwargs, window_factor


def test_uniform_FD(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    n = int(p["f_max"] / p["delta_f"]) + 1
    frequencies_expected = np.linspace(0, p["f_max"], n)
    frequencies = domain()
    assert np.linalg.norm(frequencies - frequencies_expected) < 1e-15


def test_uniform_FD_mask(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    mask = domain.frequency_mask
    n_masked = int((p["f_max"] - p["f_min"]) / p["delta_f"]) + 1
    frequencies_expected_masked = np.linspace(p["f_min"], p["f_max"], n_masked)
    frequencies_masked = domain()[mask]
    assert np.linalg.norm(frequencies_masked - frequencies_expected_masked) < 1e-15


# def test_uniform_FD_noise_std(uniform_FD_params):
#     p = uniform_FD_params
#     domain = FrequencyDomain(**p)
#     expected = np.sqrt(p['window_factor']) / np.sqrt(4.0 * p['delta_f'])
#     assert np.abs(domain.noise_std - expected) < 1e-15


def test_FD_domain_dict(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    domain2 = build_domain(domain.domain_dict)
    assert domain.__dict__ == domain2.__dict__


def test_FD_update_data(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    n = int(p["f_max"] / p["delta_f"]) + 1
    nt = int((p["f_max"] - p["f_min"]) / p["delta_f"]) + 1
    # test that the sample frequencies are correct
    assert n == len(domain) == len(domain())
    assert nt == int(np.sum(domain.frequency_mask))
    assert np.all(domain() == np.linspace(0, p["f_max"], n))
    assert np.all(
        domain()[np.nonzero(domain.frequency_mask)]
        == np.linspace(p["f_min"], p["f_max"], nt)
    )
    # Test that data range adjustment works specifying different axis
    a = np.random.rand(10, n, 20)
    b = domain.update_data(a, axis=1)
    assert np.all(b[:, : domain.min_idx, :] == 0.0)
    assert np.all(b[:, domain.min_idx : n + 1, :] == a[:, domain.min_idx :, :])
    # Test that data range adjustment works as intended with nonzero low_value
    a = np.random.random(n)
    assert np.all(domain.update_data(a) == a * domain.frequency_mask)
    assert np.all(
        domain.update_data(a, low_value=5.0)[: domain.min_idx]
        == 5.0 * np.ones(round(p["f_min"] / p["delta_f"]))
    )


def test_FD_set_new_range(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    # test that ValueErrors are raised for infeasible inputs
    with pytest.raises(ValueError):
        domain.set_new_range(p["f_max"] + 10, None)
    with pytest.raises(ValueError):
        domain.set_new_range(p["f_min"] - 10, None)
    with pytest.raises(ValueError):
        domain.set_new_range(None, p["f_max"] + 10)
    with pytest.raises(ValueError):
        domain.set_new_range(None, p["f_min"] - 10)
    with pytest.raises(ValueError):
        domain.set_new_range(p["f_min"] + 10, p["f_min"] + 5)
    # test that setting new frequency range works as intended
    f_min_new, f_max_new = 40, 800
    n = int(f_max_new / p["delta_f"]) + 1
    domain.set_new_range(f_min_new, f_max_new)
    assert n == len(domain) == len(domain())
    assert np.all(domain() == np.linspace(0, f_max_new, n))
    mask = np.ones(n)
    mask[: round(f_min_new / p["delta_f"])] = 0.0
    assert np.all(domain.frequency_mask == mask)
    # Test that data range adjustment works as intended when setting new fmin, fmax
    a = np.random.random(n)
    assert np.all(domain.update_data(a) == a * domain.frequency_mask)
    assert np.all(
        domain.update_data(a, low_value=5.0)[: domain.min_idx]
        == 5.0 * np.ones(round(f_min_new / p["delta_f"]))
    )


def test_FD_time_translation(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    # test normal time translation
    dt = 1e-3
    a = np.sin(np.outer(np.arange(3) + 1, domain()) / 100) + 1j * np.cos(
        np.outer(np.arange(3) + 1, domain()) / 100
    )
    a_dt = a * np.exp(-2j * np.pi * domain() * dt)
    b_dt = domain.time_translate_data(a, dt)
    assert np.allclose(a_dt, b_dt)
    assert not np.allclose(a_dt, domain.time_translate_data(a, 1.01 * dt))
    # test that time translation can be undone
    assert np.allclose(a, domain.time_translate_data(a_dt, -dt))
    assert not np.allclose(a, domain.time_translate_data(a_dt, -1.01 * dt))


def test_FD_time_translation_torch(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    batch_size = 5
    num_detectors = 2
    dt = torch.randn(batch_size, num_detectors, dtype=torch.float32)
    data = torch.empty(
        (batch_size, num_detectors, 3, len(domain) - domain.min_idx),
        dtype=torch.float32,
    )
    constant_value = 2.0
    f = domain.sample_frequencies_torch[domain.min_idx :]
    for i in range(batch_size):
        for j in range(num_detectors):
            signal = torch.exp(-2j * np.pi * dt[i, j] * f)
            data[i, j, 0, :] = signal.real
            data[i, j, 1, :] = signal.imag
            data[i, j, 2, :] = constant_value
    result = domain.time_translate_data(data, -dt)
    assert result.dtype == torch.float32
    assert result.shape == data.shape
    assert torch.allclose(result[..., 0, :], torch.tensor(1.0), atol=1e-5)
    # Tolerance of 1e-2 is required in the imaginary part, likely because we are
    # checking that it *vanishes*. This is a consequence of single-precision floats.
    # TODO: Is there a way to improve on this?
    assert torch.allclose(result[..., 1, :], torch.tensor(0.0), atol=1e-2)
    assert torch.allclose(result[..., 2, :], torch.tensor(constant_value))


def test_FD_caching(uniform_FD_params):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    domain_ref = FrequencyDomain(**p)

    assert np.all(domain() == domain_ref())
    # we now modify domain._f_max by hand, which should not be done, as this
    # does not update the cached properties
    domain._f_max = 50
    assert np.all(domain() == domain_ref())
    domain._reset_caches()
    # after clearing the cache, the __call__ method should return the correct
    # result
    assert np.all(domain() != domain_ref())


def test_FD_window_factor(uniform_FD_params, window_setup):
    p = uniform_FD_params
    domain = FrequencyDomain(**p)
    _, window_factor = window_setup
    assert window_factor == 0.9374713897717841
    # check that window_factor is initially None
    assert domain.window_factor is None
    # set new window_factor
    domain.window_factor = window_factor
    assert domain._window_factor == domain.window_factor == window_factor
    noise_std = domain.noise_std
    assert noise_std == np.sqrt(domain.window_factor) / np.sqrt(4 * domain.delta_f)
    window_factor = 1
    # now set new window factor correctly via the setter and check that
    # noise_std is updated as intended since the cache is cleared
    domain.window_factor = window_factor
    assert domain._window_factor == domain.window_factor == window_factor
    assert domain.noise_std != noise_std
    assert domain.noise_std == np.sqrt(domain.window_factor) / np.sqrt(
        4 * domain.delta_f
    )


# def test_FD_truncation_old(uniform_FD_params):
#     p = uniform_FD_params
#     domain = FrequencyDomain(**p)
#     N = len(domain)
#     domain.initialize_truncation((40,1024))
#     assert domain._truncated_sample_frequencies[0] == 40
#     assert domain._truncated_sample_frequencies[-1] == 1024
#     # test that array can be truncated with automatic axis selection
#     a = np.random.rand(3, N, 4, 2)
#     a_truncated = domain.truncate_data(a)
#     assert a_truncated.shape[1] == domain._truncation_num_bins
#     assert a_truncated.shape[0] == a.shape[0]
#     assert a_truncated.shape[2:] == a.shape[2:]
#     assert np.all(a[:,domain._truncation_idx_lower:domain._truncation_idx_upper]
#                   == a_truncated)
#     assert not np.all(a[:,0:domain._truncation_num_bins] == a_truncated)
#     # test that axis can be selected manually
#     a_truncated_2 = domain.truncate_data(a, axis=1)
#     assert np.all(a_truncated == a_truncated_2)
#     # test that errors are raised as intended
#     with pytest.raises(ValueError):
#         domain.truncate_data(a, axis=0)
#     with pytest.raises(ValueError):
#         domain.truncate_data(np.zeros((10, N-1, N+1)))
#     with pytest.raises(ValueError):
#         domain.truncate_data(np.zeros((10, N, N)))
#     # test that manual axis selection works in the above case
#     assert domain.truncate_data(np.zeros((10, N, N)), axis=1).shape == \
#            (10, domain._truncation_num_bins, N)

# def test_FD_time_translation_old(uniform_FD_params):
#     p = uniform_FD_params
#     domain = FrequencyDomain(**p)
#     domain.initialize_truncation((40,1024))
#     dt = 1e-3
#     data = np.sin(np.outer(np.arange(3)+1, domain())/100) + \
#            1j * np.cos(np.outer(np.arange(3)+1, domain())/100)
#     data_translated = domain.time_translate_data(data, dt)
#     # check that time translation has correct inverse
#     assert not np.allclose(data_translated, data)
#     assert np.allclose(domain.time_translate_data(data_translated, -dt), data)
#     # check that truncation and time translation commutes
#     data_truncated = domain.truncate_data(data)
#     data_truncated_translated = domain.time_translate_data(data_truncated, dt)
#     data_translated_truncated = domain.truncate_data(data_translated)
#     assert np.all(data_translated_truncated == data_truncated_translated)
#     # import matplotlib.pyplot as plt
#     # plt.plot(data[0].real)
#     # plt.plot(data_translated[0].real)
#     # plt.plot(domain.time_translate_data(data_translated, -dt)[0].real)
#     # plt.show()


# def test_TD():
#     time_duration, sampling_rate = 4.0, 1.0 / 4096.0
#     domain = TimeDomain(time_duration, sampling_rate)
#     delta_t = 1.0 / sampling_rate
#     n = time_duration / delta_t
#     times_expected = np.arange(n) * delta_t
#     assert np.linalg.norm(domain() - times_expected) < 1e-15
