from functools import partial
from multiprocessing import Pool
from itertools import starmap

import numpy as np
from bilby.core.utils import random
from scipy.stats import rv_histogram
from threadpoolctl import threadpool_limits


def interpolated_sample_and_log_prob_multi(
    sample_points, values, num_processes: int = 1
):
    """
    Given a distribution discretized on a grid, return a sample and the log prob from an
    interpolated distribution, the piecewise-constant density of
    `_cell_mean_histogram`. Works with multiprocessing.

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
    with threadpool_limits(limits=1, user_api="blas"):
        data_generator = iter(values)
        task_fun = partial(interpolated_sample_and_log_prob, sample_points)
        if num_processes > 1:
            # Workers are not re-seeded: under fork they copy one random stream,
            # under spawn they start unseeded (#408).
            with Pool(processes=num_processes) as pool:
                result_list = pool.map(task_fun, data_generator)
        else:
            result_list = list(map(task_fun, data_generator))
    sample, log_prob = np.array(result_list).T
    return sample, log_prob


def interpolated_sample_and_log_prob(sample_points, values):
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

    Returns
    -------
    (float, float) : sample and log_prob
    """
    dist = _cell_mean_histogram(sample_points, values)
    sample = dist.rvs(random_state=random.rng)
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
    with threadpool_limits(limits=1, user_api="blas"):
        data_generator = zip(iter(values), iter(evaluation_points))
        task_fun = partial(interpolated_log_prob, sample_points)
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                result_list = pool.starmap(task_fun, data_generator)
        else:
            result_list = list(starmap(task_fun, data_generator))
    return np.array(result_list)


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
