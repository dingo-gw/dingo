from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class Likelihood(object):
    def log_likelihood(self, theta, include_supplemental=False):
        raise NotImplementedError("log_likelihood() should be implemented in subclass.")

    def log_likelihood_multi(
        self,
        theta: pd.DataFrame,
        include_supplemental=False,
        num_processes: int = 1,
    ):
        """
        Calculate the log likelihood at multiple points in parameter space. Works with
        multiprocessing.

        This wraps the log_likelihood() method.

        Parameters
        ----------
        theta : pd.DataFrame
            Parameters values at which to evaluate likelihood.
        include_supplemental : bool
            Whether to return also a dictionary containing additional information associated to the likelihood
            evaluation.
        num_processes : int
            Number of processes to use.

        Returns
        -------
        (log_likelihood array, supplemental dict [optional])
        """
        partial_log_likelihood = partial(
            log_likelihood_task_func,
            include_supplemental=include_supplemental,
            likelihood=self,
        )

        with threadpool_limits(limits=1, user_api="blas"):
            # Generator object for theta rows. For idx this yields row idx of
            # theta dataframe, converted to dict, ready to be passed to
            # log_likelihood.
            theta_generator = (d[1].to_dict() for d in theta.iterrows())

            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    map_result = list(pool.map(partial_log_likelihood, theta_generator))
            else:
                map_result = list(map(partial_log_likelihood, theta_generator))

        if include_supplemental:
            log_likelihood, supplemental = zip(*map_result)
            log_likelihood = np.array(list(log_likelihood))
            supplemental = list(supplemental)
            supplemental_keys = supplemental[0].keys()
            supplemental = {
                k: np.array([s[k] for s in supplemental]) for k in supplemental_keys
            }
            return log_likelihood, supplemental
        else:
            return np.array(map_result)


def log_likelihood_task_func(
    theta: dict, include_supplemental: bool, likelihood: Likelihood
):
    return likelihood.log_likelihood(theta, include_supplemental=include_supplemental)
