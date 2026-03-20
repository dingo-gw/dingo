"""
Smoke test for LAL approximant availability using new API.

Ported from dingo-waveform tests/test_approximants.py.
"""

from dingo.gw.approximant import Approximant
from dingo.gw.prior import IntrinsicPriors
from dingo.gw.waveform_generator.new_api import build_waveform_generator


def test_IMRPhenomXPHM():
    """Check that IMRPhenomXPHM generation produces no exceptions."""
    config = {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
        },
        "waveform_generator": {
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
    }
    intrinsic_prior_dict = {
        "mass_1": 50.0,
        "mass_2": 25.0,
        "chirp_mass": 60.0,
        "mass_ratio": 0.5,
        "phase": 2.5811112632546123,
        "a_1": 0.5,
        "a_2": 0.6,
        "tilt_1": 1.8222778934660213,
        "tilt_2": 1.3641458250460199,
        "phi_12": 4.469204665688967,
        "phi_jl": 3.021398659177057,
        "theta_jn": 1.4262724019800959,
        "luminosity_distance": 100.0,
        "geocent_time": 0.0,
    }
    waveform_generator = build_waveform_generator(config)
    waveform_parameters = IntrinsicPriors(**intrinsic_prior_dict).sample()
    waveform_generator.generate_hplus_hcross(waveform_parameters)
