from functools import partial
from multiprocessing import Pool
from itertools import starmap

import numpy as np
from bilby.core.prior import Interped
from threadpoolctl import threadpool_limits


def interpolated_sample_and_log_prob_multi(
    sample_points, values, num_processes: int = 1
):
    """
    Given a distribution discretized on a grid, return a sample and the log prob from an
    interpolated distribution. Wraps the bilby.core.prior.Interped class. Works with
    multiprocessing.

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
            with Pool(processes=num_processes) as pool:
                result_list = pool.map(task_fun, data_generator)
        else:
            result_list = list(map(task_fun, data_generator))
    sample, log_prob = np.array(result_list).T
    return sample, log_prob


def interpolated_sample_and_log_prob(sample_points, values):
    """
    Given a distribution discretized on a grid, return a sample and the log prob from an
    interpolated distribution. Wraps the bilby.core.prior.Interped class.

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
    interp = Interped(sample_points, values)
    sample = interp.sample()
    log_prob = interp.ln_prob(sample)
    return sample, log_prob


def interpolated_log_prob_multi(
    sample_points, values, evaluation_points, num_processes: int = 1
):
    """
    Given a distribution discretized on a grid, the log prob at a specific point
    using an interpolated distribution. Wraps the bilby.core.prior.Interped class.
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
    interpolated distribution. Wraps the bilby.core.prior.Interped class.

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
    interp = Interped(sample_points, values)
    return interp.ln_prob(evaluation_point)
