import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.noise.synthetic.asd_parameterization import (
    curve_fit,
    fit_broadband_noise,
    parameterize_single_psd,
)
from dingo.gw.noise.synthetic.utils import (
    get_index_for_elem,
    lorentzian_eval,
    reconstruct_psds_from_parameters,
)


@pytest.fixture()
def domain():
    return UniformFrequencyDomain(f_min=0.0, f_max=1024.0, delta_f=1.0)


# ---------------------------------------------------------------------------
# utils: get_index_for_elem, lorentzian_eval
# ---------------------------------------------------------------------------


def test_get_index_for_elem_returns_nearest():
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    assert get_index_for_elem(arr, 2.0) == 2  # exact
    assert get_index_for_elem(arr, 2.4) == 2  # nearest below
    assert get_index_for_elem(arr, 2.6) == 3  # nearest above


def test_lorentzian_eval_returns_zeros_for_degenerate_params(domain):
    x = domain.sample_frequencies
    np.testing.assert_array_equal(lorentzian_eval(x, 0.0, 5.0, 100.0), np.zeros_like(x))
    np.testing.assert_array_equal(
        lorentzian_eval(x, 200.0, -1.0, 100.0), np.zeros_like(x)
    )


def test_lorentzian_eval_peaks_at_f0(domain):
    x = domain.sample_frequencies
    f0 = 200.0
    line = lorentzian_eval(x, f0, 5.0, 100.0)
    assert line.shape == x.shape
    assert line.max() > 0
    assert x[np.argmax(line)] == pytest.approx(f0, abs=domain.delta_f)


def test_lorentzian_eval_truncation_suppresses_tails(domain):
    x = domain.sample_frequencies
    f0 = 200.0
    full = lorentzian_eval(x, f0, 5.0, 100.0)
    truncated = lorentzian_eval(x, f0, 5.0, 100.0, delta_f=5.0)
    far_from_peak = np.abs(x - f0) > 50
    # The exponential truncation makes the tails strictly smaller.
    assert truncated[far_from_peak].sum() < full[far_from_peak].sum()


# ---------------------------------------------------------------------------
# fit_broadband_noise
# ---------------------------------------------------------------------------


def test_fit_broadband_noise_recovers_flat_level(domain):
    # A PSD that is constant in log space should be reproduced by the spline nodes.
    const = np.log(1e-40)
    log_psd = np.full_like(domain.sample_frequencies, const)
    xs, ys = fit_broadband_noise(domain, log_psd, num_spline_positions=8, sigma=1.0)

    assert len(xs) == len(ys) == 8
    # ys are accumulated in float32, so rounding limits agreement to ~1e-5
    np.testing.assert_allclose(ys, const, atol=1e-4)
    # Node positions are increasing and lie within the frequency range.
    assert np.all(np.diff(xs) > 0)
    assert xs[0] >= domain.sample_frequencies[0]
    assert xs[-1] <= domain.sample_frequencies[-1]


# ---------------------------------------------------------------------------
# curve_fit
# ---------------------------------------------------------------------------


def test_curve_fit_recovers_injected_lorentzian(domain):
    x = domain.sample_frequencies
    segment = (x >= 150) & (x <= 250)
    frequencies = x[segment]
    f0_true, A_true, Q_true = 200.0, 5.0, 100.0
    line = lorentzian_eval(frequencies, f0_true, A_true, Q_true)

    data = {
        "psd": line,
        "broadband_noise": np.zeros_like(frequencies),
        "frequencies": frequencies,
        "lower_freq": frequencies[0],
        "upper_freq": frequencies[-1],
    }
    f0, A, Q = curve_fit(data, std=1e-3)

    assert f0 == pytest.approx(f0_true, abs=2 * domain.delta_f)
    # Fitted parameters respect the optimizer bounds.
    assert frequencies[0] <= f0 <= frequencies[-1]
    assert 0 <= A <= 12
    assert 10 <= Q <= 1000


# ---------------------------------------------------------------------------
# parameterize_single_psd (integration)
# ---------------------------------------------------------------------------


def test_parameterize_single_psd_shapes(domain):
    psd = np.full_like(domain.sample_frequencies, 1e-40)
    settings = {
        "sigma": 1.0,
        "num_spline_positions": 8,
        "num_spectral_segments": 5,
        "delta_f": -1,  # non-positive -> no Lorentzian truncation
    }
    out = parameterize_single_psd(psd, domain, settings)

    assert set(out.keys()) == {"x_positions", "y_values", "spectral_features"}
    assert len(out["y_values"]) == settings["num_spline_positions"]
    assert len(out["x_positions"]) == settings["num_spline_positions"]
    assert out["spectral_features"].shape == (settings["num_spectral_segments"], 3)


# ---------------------------------------------------------------------------
# reconstruct_psds_from_parameters
# ---------------------------------------------------------------------------


def test_reconstruct_psds_from_parameters_shape_and_positivity(domain):
    num_spline, num_segments = 8, 5
    parameters = {
        "x_positions": np.linspace(20.0, domain.f_max, num_spline),
        "y_values": np.full(num_spline, np.log(1e-40)),
        "spectral_features": np.zeros((num_segments, 3)),  # no spectral lines
    }
    # smoothen=True -> deterministic exp(base_noise) reconstruction (no added noise).
    psds = reconstruct_psds_from_parameters(
        parameters, domain, {"sigma": 1.0, "smoothen": True}
    )
    assert psds.shape == (1, len(domain))
    assert np.all(psds > 0)
    # A cubic spline through a constant is exactly that constant, so exp(base_noise)
    # reconstructs 1e-40 to (float64) machine precision (observed rel. error ~1e-15).
    np.testing.assert_allclose(psds[0], 1e-40, rtol=1e-12)
