"""
Tests for new UniformFrequencyDomain API (window_factor, get_parameters,
from_parameters via build_domain).

Ported from dingo-waveform tests/test_frequency_domain.py.
"""

import numpy as np
import pytest
import torch
from scipy.signal.windows import tukey

from dingo.gw.domains import DomainParameters, UniformFrequencyDomain, build_domain

_uniform_FD_params = {"f_min": 20.0, "f_max": 4096.0, "delta_f": 1.0 / 4.0}


def _get_tukey_window_factor(T: float, f_s: int, roll_off: float):
    alpha = 2 * roll_off / T
    w = tukey(int(T * f_s), alpha)
    return np.sum(w**2) / len(w)


def test_FD_get_parameters_roundtrip():
    """Test roundtrip via get_parameters -> build_domain."""
    p = _uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    domain_parameters = domain.get_parameters()
    domain2 = build_domain(domain_parameters)
    d1 = domain.__dict__
    d2 = domain2.__dict__
    assert set(d1.keys()) == set(d2.keys())
    for k in d1.keys():
        assert d1[k] == d2[k]


def test_FD_window_factor():
    p = _uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    window_factor = _get_tukey_window_factor(T=8.0, f_s=4096, roll_off=0.4)
    assert window_factor == 0.9374713897717841
    # check that window_factor is initially None
    assert domain.window_factor is None
    # set new window_factor
    domain.window_factor = window_factor
    assert domain._window_factor == domain.window_factor == window_factor
    noise_std = domain.noise_std
    assert noise_std == np.sqrt(domain.window_factor) / np.sqrt(4 * domain.delta_f)
    window_factor = 1
    # now set new window factor correctly via the setter
    domain.window_factor = window_factor
    assert domain._window_factor == domain.window_factor == window_factor
    assert domain.noise_std != noise_std
    assert domain.noise_std == np.sqrt(domain.window_factor) / np.sqrt(
        4 * domain.delta_f
    )


def test_FD_window_factor_none_noise_std():
    """When window_factor is None, noise_std should use the legacy formula."""
    p = _uniform_FD_params
    domain = UniformFrequencyDomain(**p)
    # With window_factor=None, noise_std should use the base class formula
    noise_std = domain.noise_std
    expected = 1 / np.sqrt(4.0 * domain.delta_f)
    assert noise_std == expected


def test_FD_window_factor_in_constructor():
    """Test that window_factor can be set in constructor."""
    p = _uniform_FD_params
    domain = UniformFrequencyDomain(**p, window_factor=0.5)
    assert domain.window_factor == 0.5
    # noise_std should use window_factor
    expected = np.sqrt(0.5) / np.sqrt(4 * domain.delta_f)
    assert domain.noise_std == expected


def test_FD_from_parameters():
    """Test creating UFD from DomainParameters."""
    params = DomainParameters(
        f_min=20.0,
        f_max=4096.0,
        delta_f=0.25,
        type="dingo.gw.domains.uniform_frequency_domain.UniformFrequencyDomain",
    )
    domain = UniformFrequencyDomain.from_parameters(params)
    assert domain.f_min == 20.0
    assert domain.f_max == 4096.0
    assert domain.delta_f == 0.25


def test_FD_from_parameters_missing_field():
    """Test that from_parameters raises error when required field is missing."""
    params = DomainParameters(
        f_min=20.0,
        f_max=4096.0,
        # delta_f missing
        type="UniformFrequencyDomain",
    )
    with pytest.raises(ValueError, match="should not be None"):
        UniformFrequencyDomain.from_parameters(params)
