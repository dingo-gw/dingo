"""
Tests for MultibandedFrequencyDomain decimation quality using new API.

Ported from dingo-waveform tests/test_mfd_decimation.py.
Verifies that UFD waveforms decimated to MFD match directly-generated MFD waveforms.
"""

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pytest
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d

from dingo.gw.approximant import Approximant
from dingo.gw.domains import MultibandedFrequencyDomain, UniformFrequencyDomain
from dingo.gw.prior import IntrinsicPriors
from dingo.gw.waveform_generator.new_api import build_waveform_generator
from dingo.gw.waveform_generator.polarizations import Polarization


def _get_mismatch(
    a: np.ndarray,
    b: np.ndarray,
    domain,
    asd_file: Optional[str] = None,
) -> float:
    """Mismatch is 1 - overlap."""
    if asd_file is not None:
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain())
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((np.conj(a) * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((np.conj(a) * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((np.conj(b) * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


_approximants = ("IMRPhenomXPHM", "SEOBNRv4PHM")


@pytest.fixture
def mfd():
    return MultibandedFrequencyDomain(
        nodes=[20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        delta_f_initial=0.0625,
        base_delta_f=0.0625,
    )


@pytest.fixture(params=_approximants)
def approximant(request):
    return Approximant(request.param)


@pytest.fixture
def intrinsic_prior(approximant: Approximant):
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
    return IntrinsicPriors(**intrinsic_dict)


@pytest.fixture
def wfg_mfd(mfd, approximant):
    return build_waveform_generator(
        {
            "approximant": str(approximant),
            "f_ref": 10.0,
            "f_start": 10.0,
            "spin_conversion_phase": 0.0,
        },
        mfd,
    )


@pytest.fixture
def wfg_ufd(approximant, mfd):
    base_domain = UniformFrequencyDomain(delta_f=0.0625, f_min=0.0, f_max=mfd.f_max)
    return build_waveform_generator(
        {
            "approximant": str(approximant),
            "f_ref": 10.0,
            "f_start": 10.0,
            "spin_conversion_phase": 0.0,
        },
        base_domain,
    )


@pytest.fixture
def num_evaluations(approximant: Approximant):
    if approximant == Approximant("SEOBNRv4PHM"):
        return 1
    else:
        return 10


@pytest.fixture
def decimation_tolerance(approximant: Approximant) -> float:
    if approximant == Approximant("IMRPhenomXPHM"):
        return 1e-4
    else:
        return 1e-9


@pytest.mark.parametrize("approximant", _approximants)
def test_decimation_quality(
    intrinsic_prior, wfg_mfd, wfg_ufd, mfd, num_evaluations, decimation_tolerance
):
    """
    Test that decimating UFD waveforms to MFD matches directly-generated MFD waveforms.
    """
    mismatches: List[List[float]] = []

    for _ in range(num_evaluations):
        p = intrinsic_prior.sample()

        pol_mfd: Polarization = wfg_mfd.generate_hplus_hcross(p)
        pol_ufd: Polarization = wfg_ufd.generate_hplus_hcross(p)

        pol_ufd_decimated = Polarization(
            h_plus=mfd.decimate(pol_ufd.h_plus),
            h_cross=mfd.decimate(pol_ufd.h_cross),
        )

        mismatches.append(
            [
                _get_mismatch(
                    asdict(pol_mfd)[pol],
                    asdict(pol_ufd_decimated)[pol],
                    mfd,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol in ["h_plus", "h_cross"]
            ]
        )

    mismatches_arr = np.array(mismatches)
    max_mismatch = np.max(mismatches_arr)

    assert max_mismatch < decimation_tolerance, (
        f"Decimation mismatch {max_mismatch:.2e} exceeds tolerance {decimation_tolerance:.2e}"
    )
