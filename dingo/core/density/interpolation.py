import bilby
import numpy as np

# Grid values per chunk of rows in the vectorized draws (~16M): bounds their
# temporaries (densities, CDFs, comparisons) to a few hundred MB.
_CHUNK_VALUES = 2**24


def interpolated_sample_and_log_prob_multi(sample_points, values):
    """
    Given distributions discretized on a grid, return one sample per distribution and
    its log prob, from the piecewise-constant density of `_cell_mean_histogram`. The
    uniform variates for the inverse-CDF draws come from bilby's generator.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N)
        x values for samples
    values : np.ndarray, shape (B, N)
        y values for samples. The distributions do not have to be initially
        normalized, although the final log_probs will be. B = batch dimension.

    Returns
    -------
    (np.ndarray, np.ndarray) : sample and log_prob arrays, each of length B
    """
    values = np.asarray(values)
    uniforms = bilby.core.utils.random.rng.uniform(size=len(values))
    edges = np.asarray(sample_points)
    chunk = max(1, _CHUNK_VALUES // len(edges))
    samples, log_probs = [], []
    for start in range(0, max(len(values), 1), chunk):
        u = uniforms[start : start + chunk]
        pdf, cdf = _cell_mean_histogram(edges, values[start : start + chunk])
        # Inverse CDF, as scipy's ppf (np.interp(u, cdf, edges)) row by row: the last
        # edge j with cdf[j] <= u, then linear within cell j, whose mass is then
        # nonzero unless u is at or beyond the end of the CDF (handled below, as is
        # u = 0).
        j = np.minimum((cdf <= u[:, None]).sum(axis=1) - 1, len(edges) - 2)
        rows = np.arange(len(u))
        lower, upper = cdf[rows, j], cdf[rows, j + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (edges[j + 1] - edges[j]) / (upper - lower)
            sample = slope * (u - lower) + edges[j]
        sample = np.where(u >= cdf[:, -1], edges[-1], sample)
        sample = np.where(u == 0, edges[0], sample)
        samples.append(sample)
        log_probs.append(_log_pdf(edges, pdf, sample))
    return np.concatenate(samples), np.concatenate(log_probs)


def interpolated_log_prob_multi(sample_points, values, evaluation_points):
    """
    Given distributions discretized on a grid, return the log prob of each at its
    evaluation point, under the piecewise-constant density of `_cell_mean_histogram`.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N)
        x values for samples
    values : np.ndarray, shape (B, N)
        y values for samples. The distributions do not have to be initially
        normalized, although the final log_probs will be. B = batch dimension.
    evaluation_points : np.ndarray, shape (B)
        x values at which to evaluate log_prob.

    Returns
    -------
    np.ndarray : log_prob array of length B
    """
    values, evaluation_points = np.asarray(values), np.asarray(evaluation_points)
    edges = np.asarray(sample_points)
    chunk = max(1, _CHUNK_VALUES // len(edges))
    return np.concatenate(
        [
            _log_pdf(
                edges,
                _cell_mean_histogram(edges, values[start : start + chunk])[0],
                evaluation_points[start : start + chunk],
            )
            for start in range(0, max(len(values), 1), chunk)
        ]
    )


def _cell_mean_histogram(edges, values):
    """The piecewise-constant densities whose height in each grid cell is the mean of
    `values` at the cell's endpoints, normalized per row: each cell keeps the mass the
    trapezoid rule gives it. Unlike bilby's `Interped` (which evaluates the
    piecewise-linear interpolant but samples the piecewise-constant one), sampling
    and `_log_pdf` describe the same density. The arithmetic follows scipy's
    `rv_histogram(..., density=True)`. Returns the densities, shape (B, N - 1), and
    the CDFs at the edges, shape (B, N)."""
    values = np.asarray(values)
    widths = np.diff(edges)
    pdf = (values[:, :-1] + values[:, 1:]) / 2
    pdf = pdf / np.sum(pdf * widths, axis=1, keepdims=True)
    cdf = np.cumsum(pdf * widths, axis=1)
    return pdf, np.concatenate([np.zeros((len(pdf), 1)), cdf], axis=1)


def _log_pdf(edges, pdf, x):
    """Log density of each row of `pdf` at the corresponding `x`: -inf outside the
    grid and at its upper end, as for scipy's `rv_histogram`."""
    padded = np.pad(pdf, ((0, 0), (1, 1)))
    with np.errstate(divide="ignore"):
        return np.log(padded[np.arange(len(pdf)), np.searchsorted(edges, x, "right")])
