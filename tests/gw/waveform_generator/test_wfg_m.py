"""
Phase-shift round-trip test for the new-API generate_hplus_hcross_m.

Verifies that summing the m-mode contributions with a phase shift matches
generating h+/h× directly with phase + phase_shift. Ports the mismatch
quality test from tests/gw/waveform_generator/test_wfg_m.py, exercising
IMRPhenomXPHM, SEOBNRv4PHM, SEOBNRv5PHM, SEOBNRv5HM through the new
WaveformGenerator hierarchy.
"""

from dataclasses import fields

import numpy as np
import pytest

from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator.api import build_waveform_generator
from dingo.gw.waveform_generator.polarizations import Polarization, sum_contributions_m
from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters


_BBH_FIELDS = {f.name for f in fields(BBHWaveformParameters)}


def _to_bbh_params(theta: dict) -> BBHWaveformParameters:
    return BBHWaveformParameters(**{k: v for k, v in theta.items() if k in _BBH_FIELDS})


@pytest.fixture
def uniform_fd_domain():
    return build_domain(
        {
            "type": "UniformFrequencyDomain",
            "f_min": 10.0,
            "f_max": 2048.0,
            "delta_f": 0.125,
        }
    )


try:
    import pyseobnr  # noqa: F401

    _APPROXIMANTS = ["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM"]
except ImportError:
    _APPROXIMANTS = ["IMRPhenomXPHM", "SEOBNRv4PHM"]


def _intrinsic_prior(approximant: str):
    if "PHM" in approximant:
        intrinsic_dict = {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "geocent_time": 0.0,
        }
    else:
        # Aligned-spin approximants (e.g., SEOBNRv5HM) cannot take in-plane spins.
        intrinsic_dict = {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "chi_1": 'bilby.gw.prior.AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99))',
            "chi_2": 'bilby.gw.prior.AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99))',
            "geocent_time": 0.0,
        }
    return build_prior_with_defaults(intrinsic_dict)


def _num_evaluations(approximant: str) -> int:
    if approximant == "SEOBNRv4PHM":
        return 1
    return 10


def _tolerances(approximant: str):
    # Return (max, median) mismatches expected.
    if approximant == "IMRPhenomXPHM":
        return 2e-2, 1e-5
    if approximant == "SEOBNRv4PHM":
        return 5e-4, 5e-4
    if approximant in ("SEOBNRv5PHM", "SEOBNRv5HM"):
        return 1e-9, 1e-12
    return 1e-5, 1e-5


@pytest.mark.parametrize("approximant", _APPROXIMANTS)
def test_generate_hplus_hcross_m_phase_shift(approximant, uniform_fd_domain):
    """Sum-of-modes with a phase shift == direct generation with shifted phase."""
    prior = _intrinsic_prior(approximant)
    num_evaluations = _num_evaluations(approximant)
    max_tol, median_tol = _tolerances(approximant)

    wfg = build_waveform_generator(
        {
            "approximant": approximant,
            "f_ref": 10.0,
            "f_start": 10.0,
            "spin_conversion_phase": 0.0,
        },
        uniform_fd_domain,
    )

    mismatches = []
    for _ in range(num_evaluations):
        theta = prior.sample()
        phase_shift = np.random.uniform(high=2 * np.pi)

        pol_m = wfg.generate_hplus_hcross_m(_to_bbh_params(theta))
        pol: Polarization = sum_contributions_m(pol_m, phase_shift=phase_shift)
        pol_ref: Polarization = wfg.generate_hplus_hcross(
            _to_bbh_params({**theta, "phase": theta["phase"] + phase_shift})
        )

        mismatches.append(
            [
                get_mismatch(
                    pol.h_plus,
                    pol_ref.h_plus,
                    wfg.domain,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                ),
                get_mismatch(
                    pol.h_cross,
                    pol_ref.h_cross,
                    wfg.domain,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                ),
            ]
        )

    mismatches = np.array(mismatches)
    assert np.max(mismatches) < max_tol
    assert np.median(mismatches) < median_tol
