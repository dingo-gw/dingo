import numpy as np
import pytest

from dingo.gw.noise.synthetic.asd_sampling import KDE


NUM_ASDS = 50
NUM_SPLINE = 8
NUM_SEGMENTS = 5
DETECTORS = ["H1", "L1"]


def _parameter_dict():
    """Synthetic parameterization of a set of ASDs (the output format of
    parameterize_asd_dataset): spline y-values + Lorentzian spectral features."""

    def per_detector():
        return {
            "x_positions": np.linspace(20.0, 1000.0, NUM_SPLINE),
            "y_values": np.random.normal(-90.0, 1.0, size=(NUM_ASDS, NUM_SPLINE)),
            "spectral_features": np.random.normal(
                [100.0, -1.0, 500.0], [1.0, 0.1, 5.0], size=(NUM_ASDS, NUM_SEGMENTS, 3)
            ),
        }

    return {det: per_detector() for det in DETECTORS}


@pytest.fixture()
def kde():
    settings = {
        "bandwidth_spectral": 0.1,
        "bandwidth_spline": 0.1,
        "split_frequencies": [200.0, 500.0],
    }
    kde = KDE(_parameter_dict(), settings)
    kde.fit()
    return kde


def test_kde_sample_shapes(kde):
    out = kde.sample(num_samples=10)
    assert set(out.keys()) == set(DETECTORS)
    for det in DETECTORS:
        assert out[det]["spectral_features"].shape == (10, NUM_SEGMENTS, 3)
        assert out[det]["y_values"].shape == (10, NUM_SPLINE)


def test_kde_sample_rescaling_shifts_base_noise_mean(kde):
    rescaling_ys = {det: np.full(NUM_SPLINE, -85.0) for det in DETECTORS}
    out = kde.sample(num_samples=30, rescaling_ys=rescaling_ys)
    # Rescaling subtracts the per-node sample mean and adds the target, so the
    # post-rescaling per-node sample mean equals the target *exactly* (a mean-shift),
    # independent of sample size. Only floating-point error remains (observed ~1e-14).
    np.testing.assert_allclose(
        out["H1"]["y_values"].mean(axis=0), rescaling_ys["H1"], atol=1e-10
    )
