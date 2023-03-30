import pytest
import numpy as np
import pandas as pd

from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import WaveformGenerator, generate_waveforms_parallel
from dingo.gw.gwutils import get_mismatch
from dingo.gw.domains.multibanded_frequency_domain import number_of_zero_crossings

from dingo.gw.transforms import HeterodynePhase, Decimate


@pytest.fixture
def bns_setup():
    domain_settings = {
        "type": "FrequencyDomain",
        "f_min": 20.0,
        "f_max": 1024.0,
        "delta_f": 0.00390625,
    }
    prior_settings = {
        "mass_1": "bilby.core.prior.Constraint(minimum=1.0, maximum=2.5)",
        "mass_2": "bilby.core.prior.Constraint(minimum=1.0, maximum=2.5)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=1.0, maximum=2.0)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
        "phase": "default",
        "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "tilt_1": "default",
        "tilt_2": "default",
        "phi_12": "default",
        "phi_jl": "default",
        "theta_jn": "default",
        "lambda_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=5000)",
        "luminosity_distance": 100.0,
        "geocent_time": 0.0,
    }
    wfg_settings = {"approximant": "IMRPhenomPv2_NRTidal", "f_ref": 10.0}
    ufd = build_domain(domain_settings)
    prior = build_prior_with_defaults(prior_settings)
    return ufd, prior, wfg_settings


def max_mismatch(a, b, domain):
    if a.keys() != b.keys():
        raise ValueError()
    return np.max([np.max(get_mismatch(a[k], b[k], domain)) for k in a.keys()])


def max_number_of_zero_crossings(a, min_idx=0):
    return np.max(
        [np.max(number_of_zero_crossings(a[k][..., min_idx:].real)) for k in a.keys()]
    )


def test_heterodyning(bns_setup):
    ufd, prior, wfg_settings = bns_setup
    parameters = prior.sample(10)
    parameters_pd = pd.DataFrame(parameters)
    wfg = WaveformGenerator(domain=ufd, **wfg_settings)
    wfg_het = WaveformGenerator(
        domain=ufd, transform=HeterodynePhase(ufd), **wfg_settings
    )

    polarizations = generate_waveforms_parallel(wfg, parameters_pd)
    polarizations_het1 = generate_waveforms_parallel(wfg_het, parameters_pd)
    polarizations_het2 = HeterodynePhase(ufd)(
        {"waveform": polarizations, "parameters": parameters}
    )["waveform"]

    assert max_mismatch(polarizations_het1, polarizations_het2, ufd) == 0
    assert max_mismatch(polarizations, polarizations_het1, ufd) > 1e-2

    n_roots = max_number_of_zero_crossings(polarizations, ufd.min_idx)
    n_roots_het = max_number_of_zero_crossings(polarizations_het1, ufd.min_idx)
    assert n_roots > 10 * n_roots_het
