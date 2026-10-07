from functools import partial

import bilby
import numpy as np
from scipy.stats import rv_histogram

from dingo.core.multiprocessing import apply_func_with_multiprocessing


def interpolated_sample_and_log_prob_multi(
    sample_points, values, num_processes: int = 1
):
    """
    Given a distribution discretized on a grid, return a sample and the log prob from an
    interpolated distribution, the piecewise-constant density of
    `_cell_mean_histogram`. Works with multiprocessing. The uniform variates for the
    inverse-CDF draws are drawn here, from bilby's generator, so the samples are the
    same for any num_processes.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N)
        x values for samples
    values : np.ndarray, shape (B, N)
        y values for samples. The distributions do not have to be initially
        normalized, although the final log_probs will be. B = batch dimension.
    num_processes : int
        Number of parallel processes to use.

    Returns
    -------
    (np.ndarray, np.ndarray) : sample and log_prob arrays, each of length B
    """
    uniforms = bilby.core.utils.random.rng.uniform(size=len(values))
    sample, log_prob = apply_func_with_multiprocessing(
        partial(interpolated_sample_and_log_prob, sample_points),
        (values, uniforms),
        num_processes,
    ).T
    return sample, log_prob


def interpolated_sample_and_log_prob(sample_points, values, uniform):
    """
    Given a distribution discretized on a grid, return a sample and the log prob from an
    interpolated distribution, the piecewise-constant density of
    `_cell_mean_histogram`.

    Parameters
    ----------
    sample_points : np.ndarray
        x values for samples
    values : np.ndarray
        y values for samples. The distribution does not have to be initially
        normalized, although the final log_prob will be.
    uniform : float
        Uniform variate on [0, 1], mapped to the sample by the inverse CDF.

    Returns
    -------
    (float, float) : sample and log_prob
    """
    dist = _cell_mean_histogram(sample_points, values)
    sample = dist.ppf(uniform)
    return sample, dist.logpdf(sample)


def _cell_mean_histogram(sample_points, values):
    """The piecewise-constant density whose height in each grid cell is the mean of
    `values` at the cell's endpoints, normalized. Each cell keeps the mass the
    trapezoid rule gives it, and, unlike bilby's `Interped` (which evaluates the
    piecewise-linear interpolant but samples the piecewise-constant one), sampling
    and `logpdf` describe the same density."""
    values = np.asarray(values)
    return rv_histogram(
        ((values[:-1] + values[1:]) / 2, np.asarray(sample_points)), density=True
    )


def interpolated_log_prob_multi(
    sample_points, values, evaluation_points, num_processes: int = 1
):
    """
    Given a distribution discretized on a grid, the log prob at a specific point
    using an interpolated distribution, the piecewise-constant density of
    `_cell_mean_histogram`.
    Works with multiprocessing.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N)
        x values for samples
    values : np.ndarray, shape (B, N)
        y values for samples. The distributions do not have to be initially
        normalized, although the final log_probs will be. B = batch dimension.
    evaluation_points : np.ndarray, shape (B)
        x values at which to evaluate log_prob.
    num_processes : int
        Number of parallel processes to use.

    Returns
    -------
    (np.ndarray, np.ndarray) : sample and log_prob arrays, each of length B
    """
    return apply_func_with_multiprocessing(
        partial(interpolated_log_prob, sample_points),
        (values, evaluation_points),
        num_processes,
    )


def interpolated_log_prob(sample_points, values, evaluation_point):
    """
    Given a distribution discretized on a grid, return a sample and the log prob from an
    interpolated distribution, the piecewise-constant density of
    `_cell_mean_histogram`.

    Parameters
    ----------
    sample_points : np.ndarray
        x values for samples
    values : np.ndarray
        y values for samples. The distribution does not have to be initially
        normalized, although the final log_prob will be.
    evaluation_point : float
        x value at which to evaluate log_prob.

    Returns
    -------
    float : log_prob
    """
    return _cell_mean_histogram(sample_points, values).logpdf(evaluation_point)
