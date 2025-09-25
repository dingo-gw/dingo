import torch

from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_window_factor
import pytest
import numpy as np


@pytest.fixture
def mfd_params():
    domain_settings = {
        "nodes": [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        "delta_f_initial": 0.0625,
        "base_domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 2048.0,
            "delta_f": 0.0625,
        },
    }
    return domain_settings


@pytest.fixture
def mfd(mfd_params):
    return MultibandedFrequencyDomain(**mfd_params)


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


def test_mfd_len(mfd_params, mfd):
    p = mfd_params
    len_expected = 0
    nodes = p["nodes"]
    for n in range(1, len(nodes)):
        len_expected += int(
            (nodes[n] - nodes[n - 1]) / p["delta_f_initial"] / 2 ** (n - 1)
        )
    assert len(mfd) == len_expected
    assert len(mfd()) == len_expected


def test_mfd(mfd_params, mfd):
    p = mfd_params
    nodes = p["nodes"]
    num_bins_cum = 0
    for n in range(1, len(nodes)):
        num_bins = int((nodes[n] - nodes[n - 1]) / p["delta_f_initial"] / 2 ** (n - 1))
        offset = (2 ** (n - 1) - 1) * p["delta_f_initial"] / 2
        frequencies_expected = (
            np.linspace(nodes[n - 1], nodes[n], num_bins, endpoint=False) + offset
        )
        frequencies = mfd()[num_bins_cum : num_bins_cum + num_bins]
        assert np.allclose(frequencies_expected, frequencies)
        num_bins_cum += num_bins
    assert mfd.min_idx == 0
    assert mfd.max_idx == len(mfd) - 1
    assert np.all(mfd.sample_frequencies_torch.numpy() == mfd.sample_frequencies)


def test_mfd_mask(mfd):
    mask = mfd.frequency_mask
    assert len(mask) == len(mfd)
    assert np.all(mask == 1.0)


def test_mfd_domain_dict(mfd):
    domain2 = build_domain(mfd.domain_dict)
    # dicts contain arrays so use np.testing.assert_equal to compare.
    np.testing.assert_equal(mfd.__dict__, domain2.__dict__)


def test_mfd_set_new_range(mfd_params, mfd):
    domain = mfd
    # test that ValueErrors are raised for infeasible inputs
    with pytest.raises(ValueError):
        domain._set_new_range(domain.f_max + 10, None)
    with pytest.raises(ValueError):
        domain._set_new_range(domain.f_min - 10, None)
    with pytest.raises(ValueError):
        domain._set_new_range(None, domain.f_max + 10)
    with pytest.raises(ValueError):
        domain._set_new_range(None, domain.f_min - 10)
    with pytest.raises(ValueError):
        domain._set_new_range(domain.f_min + 10, domain.f_min + 5)
    with pytest.raises(ValueError):
        domain._set_new_range(domain.f_min + 0.1, None)
    with pytest.raises(ValueError):
        domain._set_new_range(None, domain.f_max - 0.1)
    # test that setting new frequency range works as intended
    f_min_new, f_max_new = 40, 800
    domain._set_new_range(f_min_new, f_max_new)
    # Test that the end nodes are reasonable.
    assert domain.nodes[0] >= f_min_new
    assert domain.nodes[0] - f_min_new < domain.delta_f[0]
    assert domain.nodes[-1] <= f_max_new
    assert f_max_new - domain.nodes[-1] < domain.delta_f[-1]
    # Test that the length is correct.
    len_new = 0
    nodes_new = domain.nodes
    delta_f_initial_new = domain.delta_f[0]
    for n in range(1, len(nodes_new)):
        len_new += int(
            (nodes_new[n] - nodes_new[n - 1]) / delta_f_initial_new / 2 ** (n - 1)
        )
    assert len_new == len(domain())
    assert len_new == len(domain)
    # Test that the boundaries are correct
    mfd_original = MultibandedFrequencyDomain(**mfd_params)
    len_original = len(mfd_original)
    assert len_original == domain._range_update_initial_length
    assert (
        domain._range_update_idx_upper - domain._range_update_idx_lower + 1 == len_new
    )
    idx_lower = 0
    while mfd_original.sample_frequencies[idx_lower] < f_min_new:
        idx_lower += 1
    assert idx_lower == domain._range_update_idx_lower
    idx_upper = mfd_original.max_idx
    while f_max_new < mfd_original.sample_frequencies[idx_upper]:
        idx_upper -= 1
    assert idx_upper == domain._range_update_idx_upper
    # Test new mask
    assert len(domain.frequency_mask) == len_new
    assert np.all(domain.frequency_mask == 1.0)
    # Test that data range adjustment works
    a = np.random.random(len_original)
    assert np.all(
        a[domain._range_update_idx_lower : domain._range_update_idx_upper + 1]
        == domain.update_data(a)
    )
    # Test that data range adjustment works specifying different axis
    a = np.random.random((20, len_original, 30))
    assert np.all(
        a[:, domain._range_update_idx_lower : domain._range_update_idx_upper + 1, :]
        == domain.update_data(a, axis=1)
    )


def test_mfd_base_domain_consistency(mfd):
    assert np.all(mfd.sample_frequencies == mfd.decimate(mfd.base_domain()))
    assert mfd.f_min == mfd.base_domain.f_min
    assert mfd.f_max in mfd.base_domain.sample_frequencies


def test_mfd_time_translation(mfd):
    domain = mfd
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


def test_mfd_time_translation_torch(mfd):
    domain = mfd
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
    assert torch.allclose(result[..., 0, :], torch.tensor(1.0), atol=1e-4)
    # Tolerance of 1e-2 is required in the imaginary part, likely because we are
    # checking that it *vanishes*. This is a consequence of single-precision floats.
    # TODO: Is there a way to improve on this?
    assert torch.allclose(result[..., 1, :], torch.tensor(0.0), atol=1e-2)
    assert torch.allclose(result[..., 2, :], torch.tensor(constant_value))


def test_mfd_window_factor(mfd, window_setup):
    domain = mfd
    _, window_factor = window_setup
    assert window_factor == 0.9374713897717841
    # check that window_factor is initially None
    assert domain.window_factor is None
    # set new window_factor
    domain.window_factor = window_factor
    assert domain.window_factor == window_factor
    assert domain.base_domain.window_factor == window_factor
    noise_std = domain.noise_std
    assert np.all(noise_std == np.sqrt(window_factor) / np.sqrt(4 * domain.delta_f))
    assert len(domain.noise_std) == len(domain)
    window_factor = 1
    # now set new window factor correctly via the setter and check that
    # noise_std is updated as intended since the cache is cleared
    domain.window_factor = window_factor
    assert domain.window_factor == window_factor
    assert np.all(domain.noise_std != noise_std)
    assert np.all(
        domain.noise_std == np.sqrt(domain.window_factor) / np.sqrt(4 * domain.delta_f)
    )
