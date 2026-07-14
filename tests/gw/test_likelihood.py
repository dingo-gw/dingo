import numpy as np
import pytest
from bilby.core.prior import Uniform
from scipy.integrate import trapezoid

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.likelihood import (
    StationaryGaussianGWLikelihood,
    inner_product,
    inner_product_complex,
)


# ---------------------------------------------------------------------------
# Pure inner-product helpers
# ---------------------------------------------------------------------------

A = np.array([1 + 2j, 3 - 1j, 0 + 1j])
B = np.array([2 + 0j, 1 + 1j, 1 - 1j])


def test_inner_product_whitened():
    result = inner_product(A, B)
    assert result == pytest.approx(np.sum(A.conj() * B).real)
    assert result == pytest.approx(3.0)  # hand-computed


def test_inner_product_min_idx_truncates_leading_bins():
    result = inner_product(A, B, min_idx=1)
    assert result == pytest.approx(np.sum((A.conj() * B)[1:]).real)
    assert result == pytest.approx(1.0)  # hand-computed


def test_inner_product_unwhitened():
    psd = np.array([1.0, 2.0, 4.0])
    delta_f = 0.5
    result = inner_product(A, B, delta_f=delta_f, psd=psd)
    assert result == pytest.approx(4 * delta_f * np.sum(A.conj() * B / psd).real)
    assert result == pytest.approx(5.5)  # hand-computed


def test_inner_product_psd_without_delta_f_raises():
    with pytest.raises(ValueError, match="delta_f and psd"):
        inner_product(A, B, psd=np.ones(3))


def test_inner_product_sums_only_axis_0():
    # A trailing axis (e.g., a phase grid) is preserved; the sum is over axis 0 only.
    a = np.ones((3, 2))
    b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = inner_product(a, b)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [9.0, 12.0])


def test_inner_product_self_is_sum_of_squared_magnitudes():
    result = inner_product(A, A)
    assert result == pytest.approx(np.sum(np.abs(A) ** 2))
    assert result >= 0


def test_inner_product_complex_retains_imaginary_part():
    # The complex variant does not take the real part.
    result = inner_product_complex(A, B)
    assert result == pytest.approx(np.sum(A.conj() * B))
    assert np.iscomplexobj(result)
    # Its real part equals the (real) inner_product.
    assert result.real == pytest.approx(inner_product(A, B))


def test_inner_product_complex_psd_without_delta_f_raises():
    with pytest.raises(ValueError, match="delta_f and psd"):
        inner_product_complex(A, B, psd=np.ones(3))


# ---------------------------------------------------------------------------
# StationaryGaussianGWLikelihood
# ---------------------------------------------------------------------------

THETA = {
    "chirp_mass": 30.0,
    "mass_ratio": 0.8,
    "chi_1": 0.1,
    "chi_2": -0.1,
    "theta_jn": 1.0,
    "phase": 1.3,
    "ra": 1.5,
    "dec": -0.3,
    "psi": 1.2,
    "luminosity_distance": 500.0,
    "geocent_time": 0.0,
}


@pytest.fixture()
def domain():
    return UniformFrequencyDomain(20.0, 256.0, delta_f=0.5)


@pytest.fixture()
def event_data(domain):
    mask = domain.frequency_mask
    waveform = {d: np.where(mask, (1.0 + 1j) * 1e-21, 0.0) for d in ("H1", "L1")}
    asds = {d: np.where(mask, 1e-21, 1.0) for d in ("H1", "L1")}
    return {"waveform": waveform, "asds": asds}


def make_likelihood(domain, event_data, **extra):
    return StationaryGaussianGWLikelihood(
        wfg_kwargs={"approximant": "IMRPhenomD", "f_ref": 20.0},
        wfg_domain=domain,
        data_domain=domain,
        event_data=event_data,
        t_ref=1126259462.4,
        **extra,
    )


def test_log_Zn_is_minus_half_inner_product_of_data(domain, event_data):
    likelihood = make_likelihood(domain, event_data)
    expected = sum(
        -0.5 * inner_product(d, d) for d in likelihood.whitened_strains.values()
    )
    assert likelihood.log_Zn == pytest.approx(expected)


def test_log_likelihood_decomposition_identity(domain, event_data):
    """log L = log_Zn + <d, mu> - 1/2 <mu, mu>."""
    likelihood = make_likelihood(domain, event_data)
    mu = likelihood.signal(dict(THETA))["waveform"]
    d = likelihood.whitened_strains
    rho2opt = sum(inner_product(m, m) for m in mu.values())
    kappa2 = sum(inner_product(di, mi) for di, mi in zip(d.values(), mu.values()))
    expected = likelihood.log_Zn + kappa2 - 0.5 * rho2opt
    assert likelihood.log_likelihood(dict(THETA)) == pytest.approx(expected)


def test_multiple_marginalizations_raise(domain, event_data):
    likelihood = make_likelihood(
        domain,
        event_data,
        time_marginalization_kwargs={"t_lower": -0.05, "t_upper": 0.05, "n_fft": 1},
        phase_marginalization_kwargs={},
    )
    with pytest.raises(NotImplementedError):
        likelihood.log_likelihood(dict(THETA))


def _brute_force_marginal_log_likelihood(plain_likelihood, key, prior, n=1000):
    """Numerically marginalize the non-marginalized likelihood over ``key``.

    Direct adaptation of bilby's marginalization-correctness check
    (bilby/test/gw/likelihood/marginalization_test.py::TestMarginalizations._template):
    evaluate the non-marginalized likelihood on a grid of the marginalized parameter
    and integrate it against the parameter's prior with the trapezoidal rule.
    """
    values = np.linspace(prior.minimum, prior.maximum, n)
    prior_values = prior.prob(values)
    ln_likes = np.array(
        [plain_likelihood._log_likelihood({**THETA, key: float(v)}) for v in values]
    )
    like = np.exp(ln_likes - ln_likes.max())
    return np.log(trapezoid(like * prior_values, values)) + ln_likes.max()


def test_phase_marginalization_matches_brute_force_integral(domain, event_data):
    """Analytic phase-marginalized likelihood matches the brute-force integral.

    Same comparison as bilby's marginalization test (``_template``), specialized to
    the phase parameter with a uniform [0, 2*pi) prior.
    """
    marginalized = make_likelihood(
        domain, event_data, phase_marginalization_kwargs={"approximation_22_mode": True}
    )._log_likelihood_phase_marginalized(dict(THETA))

    brute_force = _brute_force_marginal_log_likelihood(
        make_likelihood(domain, event_data),
        key="phase",
        prior=Uniform(minimum=0.0, maximum=2 * np.pi, name="phase"),
    )
    # The analytic (Bessel) form is exact for a (2,2)-dominated waveform, and the
    # brute-force integrand is smooth and periodic on [0, 2*pi), so the trapezoidal
    # rule is spectrally accurate -> the two agree to the integration floor
    # (observed residual ~0). (bilby's own test uses a much looser delta=0.5.)
    assert marginalized == pytest.approx(brute_force, abs=1e-3)


def test_time_marginalization_matches_brute_force_integral(domain, event_data):
    """FFT-based time-marginalized likelihood matches the brute-force integral.

    Same comparison as bilby's marginalization test (``_template``), specialized to
    the geocent_time parameter with a uniform prior over [t_lower, t_upper].
    """
    t_lower, t_upper = -0.02, 0.02
    marginalized = make_likelihood(
        domain,
        event_data,
        time_marginalization_kwargs={
            "t_lower": t_lower,
            "t_upper": t_upper,
            "n_fft": 5,
        },
    )._log_likelihood_time_marginalized(dict(THETA))

    brute_force = _brute_force_marginal_log_likelihood(
        make_likelihood(domain, event_data),
        key="geocent_time",
        prior=Uniform(minimum=t_lower, maximum=t_upper, name="geocent_time"),
    )
    # The residual is dominated by the FFT time-grid discretization (resolution
    # delta_t / n_fft = 1 / (f_max * n_fft)); observed ~0.019 for n_fft=5. The
    # tolerance is set just above that. (bilby's own test uses a looser delta=0.5.)
    assert marginalized == pytest.approx(brute_force, abs=0.05)
