import numpy as np
import pytest
from scipy.stats import rv_histogram

from dingo.core.density.interpolation import (
    interpolated_log_prob_multi,
    interpolated_sample_and_log_prob_multi,
)


# A uniform distribution on [0, W] discretized on a grid: density = 1/W, so the
# (normalized) log prob at any interior point is -log(W). The density is
# normalized internally, so the input `values` need not be normalized.
WIDTH = 2.0
SAMPLE_POINTS = np.linspace(0.0, WIDTH, 500)
UNIFORM_VALUES = np.ones_like(SAMPLE_POINTS)


def test_interpolated_log_prob_normalizes_uniform_distribution():
    log_prob = interpolated_log_prob_multi(
        SAMPLE_POINTS, UNIFORM_VALUES[None], np.array([WIDTH / 2])
    )
    assert log_prob.shape == (1,)
    # A constant density integrates exactly under the trapezoidal normalization, so
    # ln_prob = -log(W) holds to floating-point precision.
    np.testing.assert_allclose(log_prob, -np.log(WIDTH), atol=1e-10)


def test_interpolated_sample_and_log_prob_multi_shapes():
    batch_values = np.stack([UNIFORM_VALUES] * 4)
    samples, log_probs = interpolated_sample_and_log_prob_multi(
        SAMPLE_POINTS, batch_values
    )
    assert samples.shape == (4,)
    assert log_probs.shape == (4,)
    assert np.all((samples >= 0.0) & (samples <= WIDTH))


def test_matches_scipy_rv_histogram():
    """Row by row, the draws and log probs equal those of scipy's rv_histogram over
    the cell-mean density, for the same uniform variates: rough densities with
    zero-mass cells, a non-uniform grid, and points outside and at the grid ends."""
    from bilby.core.utils import random

    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 3.0, 40))
    values = rng.exponential(size=(300, 40)) * (rng.uniform(size=(300, 40)) > 0.3)
    values[:, :3] = 0.0  # leading zero-mass cells
    dists = [rv_histogram(((v[:-1] + v[1:]) / 2, x), density=True) for v in values]

    random.seed(3)
    samples, log_probs = interpolated_sample_and_log_prob_multi(x, values)
    random.seed(3)
    uniforms = random.rng.uniform(size=len(values))
    np.testing.assert_allclose(
        samples, [d.ppf(u) for d, u in zip(dists, uniforms)], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        log_probs, [d.logpdf(s) for d, s in zip(dists, samples)], rtol=1e-12
    )

    points = rng.choice(np.concatenate([x, [x[0] - 1, x[-1] + 1], x[:-1] + 0.01]), 300)
    expected = [d.logpdf(p) for d, p in zip(dists, points)]
    np.testing.assert_allclose(
        interpolated_log_prob_multi(x, values, points), expected, rtol=1e-12
    )


@pytest.mark.parametrize("u", [0.0, np.nextafter(1.0, 0.0), 1.0])
def test_draw_at_the_ends_of_the_cdf(u, monkeypatch):
    """u = 0 maps to the lower end of the grid and u at the end of the CDF to its upper
    end, as for scipy's ppf. The uniform generator never returns 1, but a row whose
    CDF ends below 1 by rounding takes the same branch."""
    from types import SimpleNamespace

    from bilby.core.utils import random

    # random.rng resolves to Generator.rng; patch that, not the module attribute.
    monkeypatch.setattr(
        random.Generator, "rng", SimpleNamespace(uniform=lambda size: np.full(size, u))
    )
    values = np.stack([UNIFORM_VALUES, UNIFORM_VALUES])
    samples, _ = interpolated_sample_and_log_prob_multi(SAMPLE_POINTS, values)
    np.testing.assert_allclose(samples, 0.0 if u == 0 else WIDTH, atol=1e-12)


def test_samples_follow_the_evaluated_density():
    """The draws come from the density that log_prob evaluates: for x ~ q on [0, 1],
    E[1 / q(x)] = 1. bilby's Interped, which samples the piecewise-constant density
    but evaluates the piecewise-linear one, gives about 1.08 on this coarse grid."""
    from bilby.core.utils import random

    random.seed(0)
    x = np.linspace(0.0, 1.0, 9)
    values = np.exp(-0.5 * ((x - 0.4) / 0.08) ** 2) + 0.05
    _, log_q = interpolated_sample_and_log_prob_multi(x, np.tile(values, (20000, 1)))
    assert abs(np.mean(np.exp(-log_q)) - 1.0) < 0.04


def test_chunked_rows_match_unchunked(monkeypatch):
    """Processing the rows in chunks (here of 7 rows) gives the same draws and log
    probs as a single chunk."""
    from bilby.core.utils import random

    import dingo.core.density.interpolation as interpolation

    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 1.0, 50)
    values = rng.exponential(size=(30, 50))
    points = rng.uniform(size=30)
    out = []
    for chunk_values in (2**24, 7 * 50):
        monkeypatch.setattr(interpolation, "_CHUNK_VALUES", chunk_values)
        random.seed(4)
        out.append(
            (
                *interpolated_sample_and_log_prob_multi(x, values),
                interpolated_log_prob_multi(x, values, points),
            )
        )
    for single, chunked in zip(*out):
        np.testing.assert_array_equal(single, chunked)
