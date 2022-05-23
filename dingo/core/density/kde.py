from functools import partial
from multiprocessing import Pool

import numpy as np
from bilby.core.prior import Interped
from threadpoolctl import threadpool_limits


class NaiveKDE:
    def __init__(self, lower, upper, n_grid):
        self.lower = lower
        self.upper = upper
        self.grid, self.step = np.linspace(
            lower, upper, n_grid, endpoint=False, retstep=True
        )
        self.grid = self.grid + self.step / 2

        self.std = 2 * self.step

    def initialize_log_prob(self, log_prob_unnorm, uniform_weight=0.1):
        self.log_prob_unnorm = log_prob_unnorm

        # compute normalized density from log_prob
        alpha = np.max(self.log_prob_unnorm, axis=1, keepdims=True)
        self.density = np.exp(self.log_prob_unnorm - alpha)
        self.density = self.density / np.sum(self.density, axis=1, keepdims=True)
        # on top of the normalized density, add a uniform distribution with a weight of
        # uniform_weight
        self.density = (1 - uniform_weight) * self.density + uniform_weight / len(
            self.grid
        )
        self.density_cumsum = np.cumsum(self.density, axis=1)

    def sample_and_log_prob(self):
        n = len(self.density)
        x = np.random.uniform(0, 1, n)[:, np.newaxis]
        indices = np.argmax(self.density_cumsum > x, axis=1)
        delta = np.random.uniform(-self.step / 2.0, self.step / 2.0, n)
        samples = self.grid[indices] + delta
        # For each sample compute the log_prob. self.density is normalized to have sum
        # 1, but for proper normalization be need sum(density_norm * self.step) = 1,
        # so we need to divide self.density by self.step for normalization.
        density_norm = self.density[np.arange(n), indices] / (self.step)
        log_prob = np.log(density_norm)
        return samples, log_prob


class PeriodicGaussianKDE:
    def __init__(self, upper, n_grid):
        self.upper = upper
        self.grid, self.step = np.linspace(
            0, self.upper, n_grid, endpoint=False, retstep=True
        )
        self.grid = self.grid + self.step / 2

        self.std = 2 * self.step

    def initialize_log_prob(self, log_prob_unnorm, uniform_weight=0.1, blur=1.0):
        self.log_prob_unnorm = log_prob_unnorm
        self.batch_size = len(log_prob_unnorm)

        # compute density
        alpha = np.max(self.log_prob_unnorm, axis=1, keepdims=True)
        weights = np.exp(self.log_prob_unnorm - alpha)
        weights /= np.sum(weights, axis=1, keepdims=True)

        # get mean and std of gaussian distribution
        self.mean, self.std = self.get_mean_and_std(weights)
        # blur std
        self.std = self.std * blur
        # initialize uniform background
        self.uniform_weight = uniform_weight

    def get_mean_and_std(self, weights):
        # compute mean and std
        x = self.grid[np.newaxis, :]
        mean_0 = average(x, weights=weights, axis=1)
        std_0 = np.sqrt(average((x - mean_0[:, np.newaxis]) ** 2, weights, axis=1))

        # the peak could be near 0 or self.upper, in which case we need to
        # shift the data to obtain a good mean and std
        offset = self.upper / 2.
        x = (self.grid[np.newaxis, :] + offset) % self.upper
        mean_1 = average(x, weights=weights, axis=1)
        std_1 = np.sqrt(average((x - mean_1[:, np.newaxis]) ** 2, weights, axis=1))
        mean_1 = mean_1 - offset

        # now choose mean according to the smaller std
        inds_1 = np.where(std_1 < std_0)[0]
        mean, std = mean_0, std_0
        mean[inds_1] = mean_1[inds_1]
        std[inds_1] = std_1[inds_1]

        return mean, std

    def sample(self):
        x = (np.random.uniform(0, 1, self.batch_size) < self.uniform_weight).astype(int)
        sample_background = np.random.uniform(0, self.upper, self.batch_size)
        sample_gaussian = np.random.normal(self.mean, self.std)
        sample = x * sample_background + (1 - x) * sample_gaussian
        # apply periodic boundary condition
        sample %= self.upper
        return sample

    def log_prob(self, sample):
        # constant contribution from uniform background
        background = 1 / self.upper
        # distance from the sample to the mean
        distance = np.abs(sample - self.mean)
        # cyclic boundary conditions
        n_cycles = 10
        distance = distance[:, np.newaxis] + np.arange(1 - n_cycles,
                                                       n_cycles) * self.upper
        gaussian = gaussian_density(distance, 0, self.std[:, np.newaxis])
        gaussian = np.sum(gaussian, axis=1)
        # density is sum of uniform background and Gaussian distribution
        x = self.uniform_weight
        density = x * background + (1 - x) * gaussian
        return np.log(density)

    def sample_and_log_prob(self):
        sample = self.sample()
        log_prob = self.log_prob(sample)
        return sample, log_prob


def average(x, weights, axis=None):
    return np.sum(x * weights, axis=axis) / np.sum(weights, axis=axis)


def gaussian_density(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


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
    phase, log_prob = np.array(result_list).T
    return phase, log_prob


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