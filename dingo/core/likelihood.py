import numpy as np
import pandas as pd

from dingo.core.multiprocessing import apply_func_with_multiprocessing


class Likelihood(object):
    def log_likelihood(self, theta):
        raise NotImplementedError("log_likelihood() should be implemented in subclass.")

    def log_likelihood_multi(
        self, theta: pd.DataFrame, num_processes: int = 1
    ) -> np.ndarray:
        """
        Calculate the log likelihood at multiple points in parameter space. Works with
        multiprocessing, reproducibly (see `apply_func_with_multiprocessing`).

        This wraps the log_likelihood() method.

        Parameters
        ----------
        theta : pd.DataFrame
            Parameters values at which to evaluate likelihood.
        num_processes : int
            Number of processes to use.

        Returns
        -------
        np.array of log likelihoods
        """
        return apply_func_with_multiprocessing(
            self.log_likelihood, theta, num_processes
        )
