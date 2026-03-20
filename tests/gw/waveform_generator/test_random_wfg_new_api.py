"""
Tests for the RandomApproximant integration with new-style API.

Ported from dingo-waveform tests/test_random_waveform_generator.py.

Verifies that:
- The factory returns the correct subclass
- Polarization generation works and produces correct shapes
- Mode-separated generation returns expected modes
- Waveforms are reproducible (deterministic)
- Amplitude scales with 1/luminosity_distance
"""

import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator.polarizations import Polarization
from dingo.gw.waveform_generator.new_api import (
    NewWaveformGenerator,
    RandomWaveformGenerator,
    build_waveform_generator,
)
from dingo.gw.waveform_generator.waveform_parameters import RandomWaveformParameters


@pytest.fixture
def domain() -> UniformFrequencyDomain:
    return UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)


@pytest.fixture
def wfg(domain):
    return build_waveform_generator(
        {"approximant": "RandomApproximant", "f_ref": 20.0}, domain
    )


@pytest.fixture
def params() -> RandomWaveformParameters:
    return RandomWaveformParameters(
        mass_1=36.0,
        mass_2=29.0,
        luminosity_distance=1000.0,
        phase=0.5,
    )


class TestFactory:
    """Test that the factory correctly creates RandomWaveformGenerator."""

    def test_returns_correct_subclass(self, wfg):
        assert isinstance(wfg, RandomWaveformGenerator)

    def test_is_waveform_generator(self, wfg):
        assert isinstance(wfg, NewWaveformGenerator)

    def test_has_modes_method(self, wfg):
        assert hasattr(wfg, "generate_hplus_hcross_m")


class TestGenerateHplusHcross:
    """Test basic polarization generation."""

    def test_returns_polarization(self, wfg, params):
        pol = wfg.generate_hplus_hcross(params)
        assert isinstance(pol, Polarization)

    def test_correct_shape(self, wfg, domain, params):
        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape == (len(domain),)
        assert pol.h_cross.shape == (len(domain),)

    def test_complex_dtype(self, wfg, params):
        pol = wfg.generate_hplus_hcross(params)
        assert np.iscomplexobj(pol.h_plus)
        assert np.iscomplexobj(pol.h_cross)

    def test_nonzero_signal(self, wfg, params):
        pol = wfg.generate_hplus_hcross(params)
        assert np.any(np.abs(pol.h_plus) > 0)
        assert np.any(np.abs(pol.h_cross) > 0)

    def test_zeros_below_fmin(self, wfg, domain, params):
        pol = wfg.generate_hplus_hcross(params)
        f = domain.sample_frequencies
        below_fmin = f < domain.f_min
        assert np.allclose(pol.h_plus[below_fmin], 0.0)


class TestGenerateHplusHcrossM:
    """Test mode-separated generation."""

    def test_returns_dict(self, wfg, params):
        pol_m = wfg.generate_hplus_hcross_m(params)
        assert isinstance(pol_m, dict)

    def test_expected_modes(self, wfg, params):
        pol_m = wfg.generate_hplus_hcross_m(params)
        assert set(pol_m.keys()) == {22, 33, 44}

    def test_mode_polarizations_correct_shape(self, wfg, domain, params):
        pol_m = wfg.generate_hplus_hcross_m(params)
        for mode, pol in pol_m.items():
            assert isinstance(pol, Polarization)
            assert pol.h_plus.shape == (len(domain),)
            assert pol.h_cross.shape == (len(domain),)

    def test_dominant_mode_is_largest(self, wfg, params):
        pol_m = wfg.generate_hplus_hcross_m(params)
        amp_22 = np.abs(pol_m[22].h_plus).max()
        amp_33 = np.abs(pol_m[33].h_plus).max()
        amp_44 = np.abs(pol_m[44].h_plus).max()
        assert amp_22 > amp_33 > amp_44

    def test_default_phase_works(self, wfg):
        """RandomWaveformParameters.phase defaults to 0.0 (never None)."""
        params_default = RandomWaveformParameters(
            mass_1=36.0,
            mass_2=29.0,
            luminosity_distance=1000.0,
        )
        pol_m = wfg.generate_hplus_hcross_m(params_default)
        assert isinstance(pol_m, dict)
        assert len(pol_m) > 0


class TestReproducibility:
    """Test that same inputs produce same outputs."""

    def test_deterministic(self, wfg, params):
        pol1 = wfg.generate_hplus_hcross(params)
        pol2 = wfg.generate_hplus_hcross(params)
        assert np.allclose(pol1.h_plus, pol2.h_plus)
        assert np.allclose(pol1.h_cross, pol2.h_cross)


class TestDistanceScaling:
    """Test that amplitude scales with 1/luminosity_distance."""

    def test_inverse_distance(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "RandomApproximant", "f_ref": 20.0}, domain
        )
        params_near = RandomWaveformParameters(
            mass_1=36.0,
            mass_2=29.0,
            luminosity_distance=100.0,
            phase=0.0,
        )
        params_far = RandomWaveformParameters(
            mass_1=36.0,
            mass_2=29.0,
            luminosity_distance=1000.0,
            phase=0.0,
        )

        pol_near = wfg.generate_hplus_hcross(params_near)
        pol_far = wfg.generate_hplus_hcross(params_far)

        # Amplitude ratio should be ~10 (distance ratio)
        ratio = np.abs(pol_near.h_plus).max() / np.abs(pol_far.h_plus).max()
        assert np.isclose(ratio, 10.0, rtol=1e-6)
