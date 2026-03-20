"""
Extended tests for time domain and mode-separated functions using the new API.

Ported from dingo-waveform tests/test_extended_coverage.py.
These tests exercise code paths not heavily covered by existing tests,
using the new-style WaveformGenerator hierarchy.
"""

import numpy as np
import pytest
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator.new_api import build_waveform_generator
from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters
from dingo.gw.waveform_generator.polarizations import Polarization


class TestTimeDomainWaveforms:
    """Test waveform generation through new-style WaveformGenerator."""

    def get_basic_params(self):
        return BBHWaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

    def test_imrphenomxphm(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = self.get_basic_params()

        pol = wfg.generate_hplus_hcross(params)
        assert isinstance(pol, Polarization)
        assert pol.h_plus.shape[0] > 0
        assert np.any(np.abs(pol.h_plus) > 0)

    def test_with_f_start(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0, "f_start": 15.0},
            domain,
        )
        params = self.get_basic_params()

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_imrphenomd(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomD", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            chi_1=0.3,
            chi_2=0.2,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_aligned_spin(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv4", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=35.0,
            mass_2=30.0,
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            chi_1=0.3,
            chi_2=-0.2,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_extreme_spins(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=150.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.95,
            a_2=0.90,
            tilt_1=1.5,
            tilt_2=1.2,
            phi_12=2.0,
            phi_jl=1.5,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_high_mass_ratio(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=50.0,
            mass_2=10.0,
            luminosity_distance=200.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

        pol = wfg.generate_hplus_hcross(params)
        assert pol.h_plus.shape[0] > 0

    def test_various_inclinations(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )

        for theta_jn in [0.1, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            params = BBHWaveformParameters(
                mass_1=30.0,
                mass_2=25.0,
                luminosity_distance=150.0,
                theta_jn=theta_jn,
                phase=0.5,
                a_1=0.3,
                a_2=0.2,
                tilt_1=0.5,
                tilt_2=0.3,
                phi_12=1.0,
                phi_jl=0.3,
                geocent_time=0.0,
            )

            pol = wfg.generate_hplus_hcross(params)
            assert pol.h_plus.shape[0] > 0


class TestModeSeparatedWaveforms:
    """Test mode-separated waveform generation with new API."""

    def get_basic_params(self):
        return BBHWaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

    def test_mode_separated_imrphenomxphm(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = self.get_basic_params()

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert isinstance(pol_m, dict)
        assert len(pol_m) > 0

        for m, pol in pol_m.items():
            assert pol.h_plus.shape[0] > 0
            assert pol.h_cross.shape[0] > 0

    def test_mode_separated_different_masses(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=40.0,
            mass_2=20.0,
            luminosity_distance=150.0,
            theta_jn=0.5,
            phase=1.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.6,
            tilt_2=0.4,
            phi_12=1.5,
            phi_jl=0.5,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_with_spin_conversion(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0, "spin_conversion_phase": 0.5},
            domain,
        )
        params = self.get_basic_params()

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_zero_spins(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=1.0,
            phase=0.5,
            a_1=0.0,
            a_2=0.0,
            tilt_1=0.0,
            tilt_2=0.0,
            phi_12=0.0,
            phi_jl=0.0,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0

    def test_mode_separated_face_on(self):
        domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
            domain,
        )
        params = BBHWaveformParameters(
            mass_1=30.0,
            mass_2=25.0,
            luminosity_distance=100.0,
            theta_jn=0.1,
            phase=0.5,
            a_1=0.3,
            a_2=0.2,
            tilt_1=0.5,
            tilt_2=0.3,
            phi_12=1.0,
            phi_jl=0.3,
            geocent_time=0.0,
        )

        pol_m = wfg.generate_hplus_hcross_m(params)
        assert len(pol_m) > 0
