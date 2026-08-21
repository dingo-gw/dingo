"""Tests for new-style prior classes.

Ported from dingo-waveform tests/test_priors.py.
"""

import numpy as np
import pytest
from bilby.gw.prior import BBHPriorDict

from dingo.gw.prior import (
    BBHExtrinsicPriorDict,
    ExtrinsicPriors,
    IntrinsicPriors,
    Priors,
    default_extrinsic_dict,
    prior_split,
)
from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters


def test_sample_and_split():
    extrinsic_priors = ExtrinsicPriors(geocent_time=0.05, psi=np.pi / 2)
    waveform_parameters = extrinsic_priors.sample()

    assert isinstance(waveform_parameters, BBHWaveformParameters)
    assert waveform_parameters.geocent_time == 0.05
    assert waveform_parameters.psi == np.pi / 2

    intrinsic_priors = IntrinsicPriors(mass_2=20.0, tilt_1=1.0)
    waveform_parameters = intrinsic_priors.sample()

    assert isinstance(waveform_parameters, BBHWaveformParameters)
    assert waveform_parameters.mass_2 == 20.0
    assert waveform_parameters.tilt_1 == 1.0

    priors = Priors(phi_12=0.1, dec=0.2)
    waveform_parameters = priors.sample()

    assert isinstance(waveform_parameters, BBHWaveformParameters)
    assert waveform_parameters.phi_12 == 0.1
    assert waveform_parameters.dec == 0.2

    intrinsic_wf, extrinsic_wf = prior_split(waveform_parameters)

    assert intrinsic_wf.phi_12 == 0.1
    assert extrinsic_wf.dec == 0.2


def test_prior_constraint():
    d = {
        "mass_1": "bilby.core.prior.Uniform(minimum=10.0, maximum=80.0)",
        "mass_2": "bilby.core.prior.Uniform(minimum=10.0, maximum=80.0)",
        "mass_ratio": "bilby.core.prior.Constraint(minimum=0.125, maximum=1.0)",
    }
    prior = BBHPriorDict(d)
    samples = prior.sample(1000)
    assert np.all(samples["mass_1"] > samples["mass_2"])

    # Same test using IntrinsicPriors as a wrapper over BBHPriorDict
    ip = IntrinsicPriors(**d)
    waveform_params = ip.samples(1000)
    assert all([wp.mass_1 > wp.mass_2 for wp in waveform_params])


def test_mean_std():
    num_samples = 100000
    eps = 0.01
    keys = ["ra", "dec", "luminosity_distance"]

    def _test_mean_std(mean_exact, std_exact, mean_approx, std_approx):
        ratios_exact = np.array(list(mean_exact.values())) / np.array(
            list(std_exact.values())
        )
        ratios_approx = np.array(list(mean_approx.values())) / np.array(
            list(std_approx.values())
        )
        assert list(mean_exact.keys()) == keys
        assert np.allclose(ratios_exact, ratios_approx, atol=eps, rtol=eps)

    prior = BBHExtrinsicPriorDict(default_extrinsic_dict)
    mean_exact, std_exact = prior.mean_std(keys)
    mean_approx, std_approx = prior.mean_std(
        keys, sample_size=num_samples, force_numerical=True
    )
    _test_mean_std(mean_exact, std_exact, mean_approx, std_approx)

    # Same test using ExtrinsicPriors as a wrapper over BBHExtrinsicPriorDict
    ep = ExtrinsicPriors(**default_extrinsic_dict)
    mean_exact, std_exact = ep.mean_std(keys)
    mean_approx, std_approx = ep.mean_std(
        keys, sample_size=num_samples, force_numerical=True
    )
    _test_mean_std(mean_exact, std_exact, mean_approx, std_approx)


def test_intrinsic_priors_default():
    """Test IntrinsicPriors with default values."""
    defaults = IntrinsicPriors.default_priors()
    assert "mass_1" in defaults
    assert "chirp_mass" in defaults
    assert "geocent_time" in defaults


def test_intrinsic_priors_sample_as_dict():
    """Test IntrinsicPriors.sample_as_dict returns a dict."""
    ip = IntrinsicPriors(mass_1=35.0, mass_2=30.0)
    d = ip.sample_as_dict()
    assert isinstance(d, dict)
    assert "mass_1" in d
    assert d["mass_1"] == 35.0


def test_priors_get_intrinsic_extrinsic():
    """Test Priors.get_intrinsic_priors / get_extrinsic_priors."""
    p = Priors(mass_1=35.0, dec=0.5, ra=1.0)
    ip = p.get_intrinsic_priors()
    ep = p.get_extrinsic_priors()
    assert isinstance(ip, IntrinsicPriors)
    assert isinstance(ep, ExtrinsicPriors)
    assert ip.mass_1 == 35.0
    assert ep.dec == 0.5
    assert ep.ra == 1.0
