import numpy as np


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


def gaussian_density(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
