import time
from os.path import split, join
import numpy as np
import pandas as pd
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
import argparse

from dingo.gw.likelihood import build_stationary_gaussian_likelihood
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults


class UnnormalizedPosteriorDensityBBH:
    """
    Implements the *unnormalized* posterior density for BBH events. This is computed
    via Bayes' theorem

            p(theta|d) = p(d|theta) * p(theta) / p(d)

    as the product of the likelihood p(d|theta) and prior p(theta), omitting the
    constant evidence p(d).
    """

    def __init__(self, likelihood, prior):
        self.likelihood = likelihood
        self.prior = prior

    def __call__(self, theta):
        return self.log_prob(theta)

    def log_prob_multiprocessing(self, theta, num_processes=1):
        """
        Compute the log_prob of theta in parallel.

        Parameters
        ----------
        theta: pd.DataFrame
            Dataframe with parameter samples theta.
        num_processes: int
            Number of processes to use.

        Returns
        -------
        log_probs: numpy.ndarray
            Array with log_probs of theta.
        """
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                # Generator object for theta rows. For idx this yields row idx of theta
                # dataframe, converted to dict, ready to be passed to self.log_prob.
                theta_generator = (d[1].to_dict() for d in theta.iterrows())
                # compute logprobs with multiprocessing
                log_probs = pool.map(self.log_prob, theta_generator)

        return np.array(log_probs)

    def log_prob(self, theta):
        try:
            log_prior = self.prior.ln_prob(theta)
            if log_prior == -np.inf:
                return -np.inf
            log_likelihood = self.likelihood.log_prob(theta)
            return log_likelihood + log_prior
        except:
            return -np.inf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute (unnormalized) posterior density for dingo samples."
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        required=True,
        help="Path to dingo samples file.",
    )
    parser.add_argument(
        "--event_dataset",
        type=str,
        default=None,
        help="Path to dataset file for GW event data.",
    )
    parser.add_argument(
        "--f_start",
        type=float,
        default=None,
        help="Lower frequency for waveform generator.",
    )
    parser.add_argument(
        "--f_end",
        type=float,
        default=None,
        help="Upper frequency for waveform generator.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes for waveform generation.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for new samples. If empty string, overwrite old sample file.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load dingo parameter samples
    theta = pd.read_pickle(args.samples_file)
    metadata = theta.attrs

    # likelihood
    likelihood = build_stationary_gaussian_likelihood(
        metadata,
        args.event_dataset,
        {"f_start": args.f_start, "f_end": args.f_end},
    )
    # prior
    intrinsic_prior = metadata["model"]["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = get_extrinsic_prior_dict(
        metadata["model"]["train_settings"]["data"]["extrinsic_prior"]
    )
    prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})
    # posterior
    posterior = UnnormalizedPosteriorDensityBBH(likelihood, prior)

    # compute posterior log_prob
    t0 = time.time()
    print(f"Computing unnormalized target posterior density for {len(theta)} samples.")
    log_probs_target = posterior.log_prob_multiprocessing(theta, args.num_processes)
    print(f"Done. This took {time.time() - t0:.2f} seconds.")

    # insert log_probs_target into theta and save the updated samples
    theta.insert(theta.shape[1], "log_probs_target", log_probs_target)
    theta.to_pickle(
        join(split(args.sample_file)[0], args.prefix + split(args.sample_file)[1])
    )


if __name__ == "__main__":
    main()
