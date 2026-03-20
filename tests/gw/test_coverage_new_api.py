"""Additional coverage tests for new-style API.

Ported from dingo-waveform tests/test_coverage_improvements.py, adapted
for the ported code in dingo-gw.
"""

import numpy as np
import pytest

from dingo.gw.approximant import Approximant
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.domains import DomainParameters
from dingo.gw.waveform_generator.polarizations import (
    BatchPolarizations,
    Polarization,
    sum_contributions_m,
)
from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters


class TestDomainEdgeCases:
    """Test domain edge cases and error handling."""

    def test_domain_parameters(self):
        params = DomainParameters(
            type="UniformFrequencyDomain",
            f_min=20.0,
            f_max=1024.0,
            delta_f=0.125,
        )
        assert params.type == "UniformFrequencyDomain"
        assert params.f_min == 20.0

    def test_frequency_domain_properties(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        assert hasattr(domain, "f_min")
        assert hasattr(domain, "f_max")
        assert hasattr(domain, "delta_f")

    def test_frequency_domain_get_parameters(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        params = domain.get_parameters()
        assert isinstance(params, DomainParameters)
        assert params.f_min == 20.0
        assert params.f_max == 1024.0

    def test_frequency_domain_get_parameters_roundtrip(self):
        from dingo.gw.domains import build_domain

        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        params = domain.get_parameters()
        domain2 = build_domain(params)
        assert len(domain) == len(domain2)
        assert domain.f_min == domain2.f_min
        assert domain.f_max == domain2.f_max


class TestPolarizations:
    """Test polarization classes."""

    def test_polarization_creation(self):
        h_plus = np.array([1.0 + 1j, 2.0 + 2j])
        h_cross = np.array([0.5 + 0.5j, 1.0 + 1j])

        pol = Polarization(h_plus=h_plus, h_cross=h_cross)
        assert np.allclose(pol.h_plus, h_plus)
        assert np.allclose(pol.h_cross, h_cross)

    def test_batch_polarizations(self):
        h_plus = np.random.randn(5, 1000) + 1j * np.random.randn(5, 1000)
        h_cross = np.random.randn(5, 1000) + 1j * np.random.randn(5, 1000)

        batch = BatchPolarizations(h_plus=h_plus, h_cross=h_cross)
        assert batch.h_plus.shape == (5, 1000)
        assert batch.h_cross.shape == (5, 1000)
        assert len(batch) == 5

    def test_batch_polarizations_num_frequency_bins(self):
        h_plus = np.random.randn(3, 500) + 1j * np.random.randn(3, 500)
        h_cross = np.random.randn(3, 500) + 1j * np.random.randn(3, 500)

        batch = BatchPolarizations(h_plus=h_plus, h_cross=h_cross)
        assert batch.num_frequency_bins == 500

    def test_sum_contributions_m_with_phase_shift(self):
        modes = {
            -2: Polarization(
                h_plus=np.array([1.0 + 0j, 2.0 + 0j]),
                h_cross=np.array([0.5 + 0j, 1.0 + 0j]),
            ),
            2: Polarization(
                h_plus=np.array([1.0 + 0j, 2.0 + 0j]),
                h_cross=np.array([0.5 + 0j, 1.0 + 0j]),
            ),
        }

        result = sum_contributions_m(modes, phase_shift=0.5)
        assert isinstance(result, Polarization)
        assert result.h_plus.shape == (2,)


class TestWaveformParameters:
    """Test WaveformParameters edge cases."""

    def test_bbh_waveform_parameters_creation(self):
        params = BBHWaveformParameters(
            mass_1=35.0,
            mass_2=30.0,
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.5,
            a_2=0.3,
            tilt_1=1.0,
            tilt_2=0.8,
            phi_12=2.0,
            phi_jl=1.5,
            geocent_time=0.0,
            ra=2.0,
            dec=0.5,
            psi=1.0,
        )
        assert params.mass_1 == 35.0
        assert params.mass_2 == 30.0
        assert params.ra == 2.0

    def test_bbh_waveform_parameters_defaults(self):
        params = BBHWaveformParameters()
        assert params.mass_1 is None
        assert params.luminosity_distance is None

    def test_bbh_waveform_parameters_partial(self):
        params = BBHWaveformParameters(mass_1=30.0, mass_2=25.0)
        assert params.mass_1 == 30.0
        assert params.mass_2 == 25.0
        assert params.phase is None


class TestApproximants:
    """Test approximant handling."""

    def test_approximant_str(self):
        approx = Approximant("IMRPhenomXPHM")
        assert str(approx) == "IMRPhenomXPHM"

    def test_approximant_different_types(self):
        approximants = [
            "IMRPhenomD",
            "IMRPhenomXPHM",
            "SEOBNRv4PHM",
            "SEOBNRv5PHM",
            "SEOBNRv5HM",
            "RandomApproximant",
        ]
        for approx_name in approximants:
            approx = Approximant(approx_name)
            assert isinstance(approx, str)

    def test_approximant_equality(self):
        a1 = Approximant("IMRPhenomD")
        a2 = Approximant("IMRPhenomD")
        assert a1 == a2
        assert a1 == "IMRPhenomD"
