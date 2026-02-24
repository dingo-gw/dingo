import pytest
import numpy as np
import pandas as pd
import torch

from dingo.gw.domains import UniformFrequencyDomain, build_domain
from dingo.gw.transforms.waveform_transforms import (
    factor_fiducial_waveform,
    HeterodynePhase,
)
from dingo.gw.transforms.gnpe_transforms import GNPEChirp
from dingo.gw.gwutils import get_mismatch


@pytest.fixture
def domain():
    return UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)


@pytest.fixture
def complex_waveform_dict(domain):
    """Dict of complex numpy waveforms keyed by detector name."""
    n = len(domain)
    rng = np.random.default_rng(42)
    return {
        "H1": rng.standard_normal(n) + 1j * rng.standard_normal(n),
        "L1": rng.standard_normal(n) + 1j * rng.standard_normal(n),
    }


@pytest.fixture
def complex_waveform_tensor(domain):
    """Complex torch tensor waveform, batched."""
    torch.manual_seed(42)
    batch = 8
    n = len(domain)
    return torch.randn(batch, n) + 1j * torch.randn(batch, n)


# --- Tests for factor_fiducial_waveform ---


def test_factor_fiducial_waveform_roundtrip_dict(domain, complex_waveform_dict):
    """Heterodyne then inverse heterodyne recovers original data (dict input)."""
    mc = 1.2
    hetero = factor_fiducial_waveform(complex_waveform_dict, domain, mc)
    recovered = factor_fiducial_waveform(hetero, domain, mc, inverse=True)
    for k in complex_waveform_dict:
        np.testing.assert_allclose(recovered[k], complex_waveform_dict[k], atol=1e-10)


def test_factor_fiducial_waveform_roundtrip_tensor(domain, complex_waveform_tensor):
    """Heterodyne then inverse heterodyne recovers original data (tensor input)."""
    mc = torch.tensor(1.2)
    hetero = factor_fiducial_waveform(complex_waveform_tensor, domain, mc)
    recovered = factor_fiducial_waveform(hetero, domain, mc, inverse=True)
    torch.testing.assert_close(recovered, complex_waveform_tensor, atol=1e-5, rtol=1e-5)


def test_factor_fiducial_waveform_roundtrip_batched_tensor(domain):
    """Roundtrip with batched chirp mass tensor."""
    torch.manual_seed(0)
    batch = 4
    n = len(domain)
    data = torch.randn(batch, n) + 1j * torch.randn(batch, n)
    mc = torch.tensor([1.0, 1.2, 1.4, 1.6])
    hetero = factor_fiducial_waveform(data, domain, mc)
    recovered = factor_fiducial_waveform(hetero, domain, mc, inverse=True)
    torch.testing.assert_close(recovered, data, atol=1e-4, rtol=1e-4)


def test_factor_fiducial_waveform_unit_norm(domain, complex_waveform_dict):
    """Verify the phase factor has unit norm (doesn't change amplitude)."""
    mc = 1.2
    hetero = factor_fiducial_waveform(complex_waveform_dict, domain, mc)
    for k in complex_waveform_dict:
        np.testing.assert_allclose(
            np.abs(hetero[k]), np.abs(complex_waveform_dict[k]), atol=1e-12
        )


def test_factor_fiducial_waveform_order_2(domain, complex_waveform_dict):
    """Verify order=2 (1PN correction) runs and differs from order=0 result."""
    mc = 1.2
    q = 0.8
    result_0 = factor_fiducial_waveform(complex_waveform_dict, domain, mc, order=0)
    result_2 = factor_fiducial_waveform(
        complex_waveform_dict, domain, mc, mass_ratio=q, order=2
    )
    # They should differ
    for k in complex_waveform_dict:
        assert not np.allclose(result_0[k], result_2[k]), (
            "order=0 and order=2 results should differ"
        )
    # But amplitudes should still be preserved
    for k in complex_waveform_dict:
        np.testing.assert_allclose(
            np.abs(result_2[k]), np.abs(complex_waveform_dict[k]), atol=1e-12
        )


def test_factor_fiducial_waveform_invalid_order(domain, complex_waveform_dict):
    """Verify invalid order raises ValueError."""
    with pytest.raises(ValueError, match="Order 1 invalid"):
        factor_fiducial_waveform(complex_waveform_dict, domain, 1.2, order=1)


# --- Tests for HeterodynePhase ---


def test_heterodyne_phase_transform(domain, complex_waveform_dict):
    """Test HeterodynePhase.__call__ with a sample dict."""
    transform = HeterodynePhase(domain, order=0)
    sample = {
        "waveform": complex_waveform_dict,
        "parameters": {"chirp_mass": 1.2},
    }
    result = transform(sample)
    # Waveform should be modified
    for k in complex_waveform_dict:
        assert not np.allclose(result["waveform"][k], complex_waveform_dict[k])
    # Amplitudes preserved
    for k in complex_waveform_dict:
        np.testing.assert_allclose(
            np.abs(result["waveform"][k]),
            np.abs(complex_waveform_dict[k]),
            atol=1e-12,
        )


def test_heterodyne_phase_fixed_parameters(domain, complex_waveform_dict):
    """Test HeterodynePhase with fixed_parameters overriding sample parameters."""
    mc_fixed = 1.4
    transform = HeterodynePhase(
        domain, order=0, fixed_parameters={"chirp_mass": mc_fixed}
    )
    sample = {
        "waveform": complex_waveform_dict,
        "parameters": {"chirp_mass": 999.0},  # should be ignored
    }
    result = transform(sample)

    # Compare with direct call using fixed chirp mass
    expected = factor_fiducial_waveform(complex_waveform_dict, domain, mc_fixed)
    for k in complex_waveform_dict:
        np.testing.assert_allclose(result["waveform"][k], expected[k], atol=1e-12)


# --- Tests for GNPEChirp ---


def test_gnpe_chirp_training(domain, complex_waveform_dict):
    """Test GNPEChirp in training mode: proxies sampled, delta stored, waveform modified."""
    kernel = {
        "chirp_mass": "bilby.core.prior.Uniform(minimum=-0.01, maximum=0.01)",
    }
    transform = GNPEChirp(kernel, domain, order=0, inference=False)

    mc_true = 1.2
    sample = {
        "waveform": complex_waveform_dict,
        "parameters": {"chirp_mass": mc_true},
        "extrinsic_parameters": {"chirp_mass": mc_true},
    }
    result = transform(sample)
    ep = result["extrinsic_parameters"]

    # Proxy should exist and be close to true value
    assert "chirp_mass_proxy" in ep
    assert abs(ep["chirp_mass_proxy"] - mc_true) <= 0.01

    # Delta should be chirp_mass - proxy
    assert "delta_chirp_mass" in ep
    np.testing.assert_allclose(
        ep["delta_chirp_mass"], mc_true - ep["chirp_mass_proxy"]
    )

    # Waveform should be modified
    for k in complex_waveform_dict:
        assert not np.allclose(result["waveform"][k], complex_waveform_dict[k])


def test_gnpe_chirp_inference(domain):
    """Test GNPEChirp in inference mode with pre-existing proxies."""
    kernel = {
        "chirp_mass": "bilby.core.prior.Uniform(minimum=-0.01, maximum=0.01)",
    }
    transform = GNPEChirp(kernel, domain, order=0, inference=True)

    batch = 4
    n = len(domain)
    torch.manual_seed(0)
    waveform = torch.randn(batch, n) + 1j * torch.randn(batch, n)

    mc_true = torch.full((batch,), 1.2)
    mc_proxy = torch.full((batch,), 1.19)

    sample = {
        "waveform": waveform,
        "extrinsic_parameters": {
            "chirp_mass": mc_true,
            "chirp_mass_proxy": mc_proxy,
        },
    }
    result = transform(sample)
    ep = result["extrinsic_parameters"]

    # Should reuse existing proxy
    torch.testing.assert_close(ep["chirp_mass_proxy"], mc_proxy)


def test_gnpe_chirp_context_parameters(domain):
    """Verify context_parameters list is correct (should be proxy names from GNPEBase)."""
    kernel = {
        "chirp_mass": "bilby.core.prior.Uniform(minimum=-0.01, maximum=0.01)",
    }
    transform = GNPEChirp(kernel, domain, order=0)
    assert transform.context_parameters == ["chirp_mass_proxy"]


def test_gnpe_chirp_no_waveform(domain):
    """Verify GNPEChirp works when no 'waveform' key (standardization case)."""
    kernel = {
        "chirp_mass": "bilby.core.prior.Uniform(minimum=-0.01, maximum=0.01)",
    }
    transform = GNPEChirp(kernel, domain, order=0)

    sample = {
        "parameters": {"chirp_mass": 1.2},
        "extrinsic_parameters": {"chirp_mass": 1.2},
    }
    result = transform(sample)
    ep = result["extrinsic_parameters"]

    assert "chirp_mass_proxy" in ep
    assert "delta_chirp_mass" in ep
    assert "waveform" not in result


# --- Integration test with real BNS waveforms (from Max's branch) ---


def _number_of_zero_crossings(x):
    """Count zero crossings along the last axis."""
    return np.sum(np.diff(np.sign(x), axis=-1) != 0, axis=-1)


def _max_zero_crossings(polarizations, min_idx=0):
    return np.max(
        [
            np.max(_number_of_zero_crossings(v[..., min_idx:].real))
            for v in polarizations.values()
        ]
    )


def _max_mismatch(a, b, domain):
    if a.keys() != b.keys():
        raise ValueError()
    return np.max([np.max(get_mismatch(a[k], b[k], domain)) for k in a.keys()])


@pytest.mark.slow
def test_heterodyning_bns_waveforms():
    """
    Integration test: generate BNS waveforms, heterodyne them, and verify that
    (1) heterodyning is invertible (zero mismatch on roundtrip),
    (2) heterodyned waveforms have significantly fewer zero crossings.

    Adapted from Max Dax's test on the bns_add_dingo_pipe_max branch.
    """
    from dingo.gw.prior import build_prior_with_defaults
    from dingo.gw.waveform_generator import (
        WaveformGenerator,
        generate_waveforms_parallel,
    )

    domain_settings = {
        "type": "UniformFrequencyDomain",
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
    wfg = WaveformGenerator(domain=ufd, **wfg_settings)

    parameters = prior.sample(10)
    polarizations = generate_waveforms_parallel(wfg, pd.DataFrame(parameters))

    # Heterodyne and inverse heterodyne
    polarizations_het = HeterodynePhase(ufd)(
        {"waveform": polarizations, "parameters": parameters}
    )["waveform"]
    polarizations_roundtrip = HeterodynePhase(ufd, inverse=True)(
        {"waveform": polarizations_het, "parameters": parameters}
    )["waveform"]

    # Roundtrip should recover original waveforms to machine precision.
    assert _max_mismatch(polarizations, polarizations_roundtrip, ufd) < 1e-14
    # Heterodyned waveforms should differ from originals.
    assert _max_mismatch(polarizations, polarizations_het, ufd) > 1e-2

    # Heterodyned waveforms should have far fewer zero crossings (the whole point).
    n_roots = _max_zero_crossings(polarizations, ufd.min_idx)
    n_roots_het = _max_zero_crossings(polarizations_het, ufd.min_idx)
    assert n_roots > 10 * n_roots_het
