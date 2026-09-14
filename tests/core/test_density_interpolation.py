import numpy as np

from dingo.core.density.interpolation import (
    interpolated_log_prob,
    interpolated_log_prob_multi,
    interpolated_sample_and_log_prob,
    interpolated_sample_and_log_prob_multi,
)


# A uniform distribution on [0, W] discretized on a grid: density = 1/W, so the
# (normalized) log prob at any interior point is -log(W). The Interped wrapper
# normalizes internally, so the input `values` need not be normalized.
WIDTH = 2.0
SAMPLE_POINTS = np.linspace(0.0, WIDTH, 500)
UNIFORM_VALUES = np.ones_like(SAMPLE_POINTS)


def test_interpolated_log_prob_normalizes_uniform_distribution():
    log_prob = interpolated_log_prob(SAMPLE_POINTS, UNIFORM_VALUES, WIDTH / 2)
    assert log_prob == np.float64(log_prob)  # scalar
    # A constant density integrates exactly under the trapezoidal normalization, so
    # ln_prob = -log(W) holds to floating-point precision (observed residual 0.0).
    np.testing.assert_allclose(log_prob, -np.log(WIDTH), atol=1e-10)


def test_interpolated_sample_and_log_prob_in_range_and_consistent():
    np.random.seed(0)
    sample, log_prob = interpolated_sample_and_log_prob(SAMPLE_POINTS, UNIFORM_VALUES)
    assert 0.0 <= sample <= WIDTH
    # Both calls build the same Interped from the same data, so the returned log_prob
    # equals interpolated_log_prob at the drawn sample to floating-point precision.
    np.testing.assert_allclose(
        log_prob,
        interpolated_log_prob(SAMPLE_POINTS, UNIFORM_VALUES, sample),
        atol=1e-10,
    )


def test_interpolated_log_prob_multi_matches_single():
    batch_values = np.stack([UNIFORM_VALUES, UNIFORM_VALUES])
    eval_points = np.array([0.5, 1.5])
    result = interpolated_log_prob_multi(
        SAMPLE_POINTS, batch_values, eval_points, num_processes=1
    )
    assert result.shape == (2,)
    expected = [
        interpolated_log_prob(SAMPLE_POINTS, UNIFORM_VALUES, p) for p in eval_points
    ]
    np.testing.assert_allclose(result, expected)


def test_interpolated_sample_and_log_prob_multi_shapes():
    np.random.seed(0)
    batch_values = np.stack([UNIFORM_VALUES] * 4)
    samples, log_probs = interpolated_sample_and_log_prob_multi(
        SAMPLE_POINTS, batch_values, num_processes=1
    )
    assert samples.shape == (4,)
    assert log_probs.shape == (4,)
    assert np.all((samples >= 0.0) & (samples <= WIDTH))
