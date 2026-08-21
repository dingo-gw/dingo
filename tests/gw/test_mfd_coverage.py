"""
Tests for new MultibandedFrequencyDomain API (base_delta_f, get_parameters,
from_parameters, narrowed, adapt_data).

Ported from dingo-waveform tests/test_multibanded_frequency_domain.py.
"""

import numpy as np
import pytest
import torch

from dingo.gw.domains import (
    DomainParameters,
    MultibandedFrequencyDomain,
    build_domain,
    decimate_uniform,
)
from dingo.gw.domains.multibanded_frequency_domain import adapt_data

# Standard parameters for testing (using base_delta_f API)
_base_domain_params = {"f_min": 20.0, "f_max": 1024.0, "delta_f": 0.125}
_nodes = [20.0, 64.0, 256.0, 1024.0]
_delta_f_initial = 0.125


@pytest.fixture
def multibanded_domain():
    """Create a standard multibanded frequency domain using base_delta_f API."""
    return MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )


def test_multibanded_creation(multibanded_domain):
    assert multibanded_domain.num_bands == 3
    assert len(multibanded_domain) == 2656
    assert multibanded_domain.f_min == 20.0
    assert np.isclose(multibanded_domain.f_max, 1023.875)


def test_multibanded_bands_structure(multibanded_domain):
    assert np.isclose(multibanded_domain._binning.delta_f_bands[0], 0.125)
    assert np.isclose(multibanded_domain._binning.delta_f_bands[1], 0.25)
    assert np.isclose(multibanded_domain._binning.delta_f_bands[2], 0.5)

    assert multibanded_domain._binning.decimation_factors_bands[0] == 1
    assert multibanded_domain._binning.decimation_factors_bands[1] == 2
    assert multibanded_domain._binning.decimation_factors_bands[2] == 4


def test_multibanded_sample_frequencies(multibanded_domain):
    freqs = multibanded_domain.sample_frequencies
    assert len(freqs) == len(multibanded_domain)
    assert freqs[0] >= multibanded_domain.f_min
    assert freqs[-1] <= multibanded_domain.f_max
    assert np.all(np.diff(freqs) > 0)


def test_multibanded_delta_f_array(multibanded_domain):
    delta_f = multibanded_domain.delta_f
    assert isinstance(delta_f, np.ndarray)
    assert len(delta_f) == len(multibanded_domain)
    assert np.isclose(delta_f.min(), 0.125)
    assert np.isclose(delta_f.max(), 0.5)


def test_multibanded_noise_std(multibanded_domain):
    noise_std = multibanded_domain.noise_std
    assert isinstance(noise_std, np.ndarray)
    assert len(noise_std) == len(multibanded_domain)
    expected = 1.0 / np.sqrt(4.0 * multibanded_domain.delta_f)
    assert np.allclose(noise_std, expected)


def test_multibanded_frequency_mask(multibanded_domain):
    mask = multibanded_domain.frequency_mask
    assert isinstance(mask, np.ndarray)
    assert len(mask) == len(multibanded_domain)
    assert np.all(mask == 1.0)
    assert multibanded_domain.frequency_mask_length == len(multibanded_domain)


def test_multibanded_indices(multibanded_domain):
    assert multibanded_domain.min_idx == 0
    assert multibanded_domain.max_idx == len(multibanded_domain) - 1


def test_multibanded_duration_sampling_rate_not_implemented(multibanded_domain):
    with pytest.raises(NotImplementedError):
        _ = multibanded_domain.duration
    with pytest.raises(NotImplementedError):
        _ = multibanded_domain.sampling_rate


def test_decimate_uniform_numpy():
    data = np.arange(100, dtype=np.float32)
    decimated = decimate_uniform(data, 2, policy="mean")
    assert len(decimated) == 50
    assert np.isclose(decimated[0], 0.5)
    assert np.isclose(decimated[1], 2.5)

    data = np.arange(100, dtype=np.float32)
    decimated = decimate_uniform(data, 5, policy="mean")
    assert len(decimated) == 20
    assert np.isclose(decimated[0], 2.0)


def test_decimate_uniform_torch():
    data = torch.arange(100, dtype=torch.float32)
    decimated = decimate_uniform(data, 2, policy="mean")
    assert len(decimated) == 50
    assert torch.isclose(decimated[0], torch.tensor(0.5))
    assert torch.isclose(decimated[1], torch.tensor(2.5))


def test_decimate_uniform_drops_remainder():
    data = np.arange(100, dtype=np.float32)
    decimated = decimate_uniform(data, 3, policy="pick")
    # 100 // 3 = 33 bins
    assert len(decimated) == 33


def test_multibanded_decimate_numpy(multibanded_domain):
    np.random.seed(42)
    coverage = int(
        multibanded_domain._binning.nodes_indices[-1]
        - multibanded_domain._binning.nodes_indices[0]
    )
    data = np.random.randn(coverage) + 1j * np.random.randn(coverage)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)
    assert decimated.dtype == data.dtype


def test_multibanded_decimate_torch(multibanded_domain):
    torch.manual_seed(42)
    coverage = int(
        multibanded_domain._binning.nodes_indices[-1]
        - multibanded_domain._binning.nodes_indices[0]
    )
    data = torch.randn(coverage, dtype=torch.complex64)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (len(multibanded_domain),)
    assert decimated.dtype == data.dtype


def test_multibanded_decimate_batched(multibanded_domain):
    np.random.seed(42)
    batch_size = 5
    num_detectors = 3
    coverage = int(
        multibanded_domain._binning.nodes_indices[-1]
        - multibanded_domain._binning.nodes_indices[0]
    )
    data = np.random.randn(batch_size, num_detectors, coverage)

    decimated = multibanded_domain.decimate(data)
    assert decimated.shape == (batch_size, num_detectors, len(multibanded_domain))


def test_multibanded_decimate_invalid_shape(multibanded_domain):
    data = np.random.randn(100)
    with pytest.raises((IndexError, ValueError)):
        multibanded_domain.decimate(data)


def test_multibanded_get_parameters(multibanded_domain):
    params = multibanded_domain.get_parameters()
    assert isinstance(params, DomainParameters)
    assert params.type == "dingo.gw.domains.multibanded_frequency_domain.MultibandedFrequencyDomain"
    assert params.nodes == _nodes
    assert np.isclose(params.delta_f_initial, _delta_f_initial)
    assert np.isclose(params.base_delta_f, _base_domain_params["delta_f"])
    assert params.window_factor is None


def test_multibanded_from_parameters(multibanded_domain):
    params = multibanded_domain.get_parameters()
    reconstructed = MultibandedFrequencyDomain.from_parameters(params)

    assert len(reconstructed) == len(multibanded_domain)
    assert reconstructed.num_bands == multibanded_domain.num_bands
    assert np.allclose(reconstructed.nodes, multibanded_domain.nodes)
    assert np.isclose(reconstructed.f_min, multibanded_domain.f_min)
    assert np.isclose(reconstructed.f_max, multibanded_domain.f_max)


def test_multibanded_from_parameters_missing_fields():
    params = DomainParameters(
        type="MultibandedFrequencyDomain",
        nodes=[20.0, 64.0, 256.0, 1024.0],
    )
    with pytest.raises(ValueError, match="should not be None"):
        MultibandedFrequencyDomain.from_parameters(params)


def test_multibanded_build_domain_from_dict():
    domain_dict = {
        "type": "MultibandedFrequencyDomain",
        "nodes": [20.0, 64.0, 256.0, 1024.0],
        "delta_f_initial": 0.125,
        "base_delta_f": 0.125,
    }
    domain = build_domain(domain_dict)
    assert isinstance(domain, MultibandedFrequencyDomain)
    assert len(domain) == 2656
    assert domain.num_bands == 3


def test_multibanded_build_domain_from_parameters(multibanded_domain):
    params = multibanded_domain.get_parameters()
    domain = build_domain(params)
    assert isinstance(domain, MultibandedFrequencyDomain)
    assert len(domain) == len(multibanded_domain)


def test_multibanded_narrowed_basic():
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )
    original_len = len(mfd)

    narrowed_mfd = mfd.narrowed(f_min=64.0, f_max=512.0)

    assert len(mfd) == original_len
    assert len(narrowed_mfd) < original_len
    assert narrowed_mfd.f_min >= 64.0
    assert narrowed_mfd.f_max <= 512.0


def test_multibanded_narrowed_validation():
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )

    with pytest.raises(ValueError, match="f_min must be strictly smaller"):
        mfd.narrowed(f_min=100.0, f_max=50.0)

    with pytest.raises(ValueError, match="not in"):
        mfd.narrowed(f_min=10.0)

    with pytest.raises(ValueError, match="not in"):
        mfd.narrowed(f_max=2000.0)


def test_multibanded_adapt_data_basic():
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )
    original_len = len(mfd)
    data_original = np.random.randn(original_len)

    narrowed_mfd = mfd.narrowed(f_min=64.0, f_max=512.0)
    new_len = len(narrowed_mfd)

    data_adapted = adapt_data(mfd, narrowed_mfd, data_original)

    assert len(data_adapted) == new_len
    assert len(data_adapted) < len(data_original)


def test_multibanded_adapt_data_multidimensional():
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )
    original_len = len(mfd)
    data = np.random.randn(5, 3, original_len)

    narrowed_mfd = mfd.narrowed(f_min=64.0, f_max=512.0)
    new_len = len(narrowed_mfd)

    adapted = adapt_data(mfd, narrowed_mfd, data, axis=-1)
    assert adapted.shape == (5, 3, new_len)

    adapted = adapt_data(mfd, narrowed_mfd, data, axis=2)
    assert adapted.shape == (5, 3, new_len)


def test_multibanded_adapt_data_incompatible_domains():
    mfd1 = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )

    mfd2 = MultibandedFrequencyDomain(
        nodes=[10.0, 50.0, 200.0],
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
    )

    data = np.random.randn(len(mfd1))

    with pytest.raises(ValueError):
        adapt_data(mfd1, mfd2, data)


def test_multibanded_time_translate_numpy(multibanded_domain):
    np.random.seed(42)
    data = np.random.randn(len(multibanded_domain)) + 1j * np.random.randn(
        len(multibanded_domain)
    )
    dt = 0.001

    translated = multibanded_domain.time_translate_data(data, dt)
    assert translated.shape == data.shape
    assert translated.dtype == data.dtype

    back = multibanded_domain.time_translate_data(translated, -dt)
    assert np.allclose(back, data)


def test_multibanded_time_translate_torch(multibanded_domain):
    torch.manual_seed(42)
    batch_size = 5
    dt = torch.randn(batch_size, dtype=torch.float32) * 0.01
    data = torch.randn(batch_size, len(multibanded_domain), dtype=torch.complex64)

    translated = multibanded_domain.time_translate_data(data, dt)
    assert translated.shape == data.shape
    assert translated.dtype == data.dtype


def test_multibanded_with_window_factor():
    mfd = MultibandedFrequencyDomain(
        nodes=_nodes,
        delta_f_initial=_delta_f_initial,
        base_delta_f=_base_domain_params["delta_f"],
        window_factor=0.5,
    )
    assert mfd.window_factor == 0.5
    assert len(mfd) == 2656


def test_multibanded_invalid_nodes_shape():
    invalid_nodes = [[20.0, 64.0], [256.0, 1024.0]]
    with pytest.raises(ValueError, match="Expected 1D nodes array"):
        MultibandedFrequencyDomain(
            nodes=invalid_nodes,
            delta_f_initial=_delta_f_initial,
            base_delta_f=_base_domain_params["delta_f"],
        )


def test_multibanded_endpoints():
    mfd = MultibandedFrequencyDomain(
        nodes=[20.0, 64.0, 256.0, 1024.0],
        delta_f_initial=0.125,
        base_delta_f=0.125,
    )
    assert mfd.f_min == 20.0
    assert abs(mfd.f_max - 1024.0) < 1.0


def test_multibanded_efficient_representation():
    base_delta_f = 0.0625
    f_min = 20.0
    f_max = 2048.0
    nodes = [20.0, 128.0, 512.0, 2048.0]
    mfd = MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=base_delta_f, base_delta_f=base_delta_f
    )

    uniform_len = int((f_max - f_min) / base_delta_f)

    assert len(mfd) < uniform_len
    reduction = 1 - len(mfd) / uniform_len
    assert reduction > 0.3
