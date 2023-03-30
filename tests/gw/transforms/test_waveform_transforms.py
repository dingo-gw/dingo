import pytest
import numpy as np
import pandas as pd
from torchvision.transforms import Compose

from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import WaveformGenerator, generate_waveforms_parallel
from dingo.gw.gwutils import get_mismatch
from dingo.gw.domains.multibanded_frequency_domain import (
    MultibandedFrequencyDomain,
    number_of_zero_crossings,
)
from dingo.gw.dataset import generate_parameters_and_polarizations

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

    # build waveform generators and multibanded frequency domain
    wfg = WaveformGenerator(domain=ufd, **wfg_settings)
    wfg_het = WaveformGenerator(
        domain=ufd, transform=HeterodynePhase(ufd), **wfg_settings
    )
    _, pols_het = generate_parameters_and_polarizations(wfg_het, prior, 10, 0)
    mfd = MultibandedFrequencyDomain.init_from_polarizations(
        ufd, pols_het, num_bins_per_period=16, delta_f_max=2.0
    )
    wfg_het_mfd = WaveformGenerator(
        domain=mfd, transform=HeterodynePhase(mfd), **wfg_settings
    )
    wfg_het_dec = WaveformGenerator(
        domain=ufd,
        transform=Compose([HeterodynePhase(ufd), Decimate(mfd)]),
        **wfg_settings
    )

    return ufd, mfd, prior, wfg, wfg_het, wfg_het_mfd, wfg_het_dec


def max_mismatch(a, b, domain):
    if a.keys() != b.keys():
        raise ValueError()
    return np.max([np.max(get_mismatch(a[k], b[k], domain)) for k in a.keys()])


def max_number_of_zero_crossings(a, min_idx=0):
    return np.max(
        [np.max(number_of_zero_crossings(a[k][..., min_idx:].real)) for k in a.keys()]
    )


def test_heterodyning(bns_setup):
    ufd, _, prior, wfg, wfg_het, _, _ = bns_setup
    parameters = prior.sample(10)

    polarizations = generate_waveforms_parallel(wfg, pd.DataFrame(parameters))
    polarizations_het1 = generate_waveforms_parallel(wfg_het, pd.DataFrame(parameters))
    polarizations_het2 = HeterodynePhase(ufd)(
        {"waveform": polarizations, "parameters": parameters}
    )["waveform"]

    assert max_mismatch(polarizations_het1, polarizations_het2, ufd) == 0
    assert max_mismatch(polarizations, polarizations_het1, ufd) > 1e-2

    n_roots = max_number_of_zero_crossings(polarizations, ufd.min_idx)
    n_roots_het = max_number_of_zero_crossings(polarizations_het1, ufd.min_idx)
    assert n_roots > 10 * n_roots_het


def test_mfd_decimation(bns_setup):
    ufd, mfd, prior, wfg, wfg_het, wfg_het_mfd, wfg_het_dec = bns_setup
    parameters = pd.DataFrame(prior.sample(10))
    pols_het = generate_waveforms_parallel(wfg_het, parameters)
    pols_het_mfd = generate_waveforms_parallel(wfg_het_mfd, parameters)
    pols_het_dec1 = {pol_name: mfd.decimate(pol) for pol_name, pol in pols_het.items()}
    pols_het_dec2 = generate_waveforms_parallel(wfg_het_dec, parameters)

    assert max_mismatch(pols_het_dec1, pols_het_dec2, mfd) == 0
    assert max_mismatch(pols_het_dec1, pols_het_mfd, mfd) < 1e-5