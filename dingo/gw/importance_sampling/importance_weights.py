"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""
import time
import yaml
from os import rename, makedirs
from os.path import dirname, join, isfile, exists
import numpy as np
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
import argparse

from dingo.core.models import PosteriorModel
from dingo.gw.likelihood import build_stationary_gaussian_likelihood
from dingo.core.density import train_unconditional_density_estimator
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sampling-importance-resampling (SIR) for dingo models."
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Path to settings file",
    )
    parser.add_argument(
        "--event_dataset",
        type=str,
        default=None,
        help="Path to dataset file for GW event data.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for the unconditional nde and SIR results.",
    )
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = dirname(args.settings)

    return args


def get_samples_and_log_probs_from_proposal(nde, num_samples):
    """
    Generate num_samples samples from the proposal distribution, which is represented
    by an unconditional nde.

    Parameters
    ----------
    nde: dingo.core.models.PosteriorModel
        Unconditional nde used as proposal distribution.
    num_samples: int
        Number of samples to generate.

    Returns
    -------
    samples: pd.DataFrame
        Dataframe with samples from proposal distribution.
    log_probs: numpy.ndarray
        Array with log_probs of the samples.
    """
    nde.model.eval()
    with torch.no_grad():
        theta = nde.model.sample(num_samples=num_samples)
        log_probs_proposal = nde.model.log_prob(theta).cpu().numpy()

    # undo standardization
    mean, std = nde.metadata["train_settings"]["data"]["standardization"].values()
    mean = np.array([v for v in mean.values()])
    std = np.array([v for v in std.values()])
    theta = theta.cpu().numpy() * std + mean

    # convert to pd.DataFrame
    columns = nde.metadata["train_settings"]["data"]["standardization"]["mean"].keys()
    theta = pd.DataFrame(theta, columns=columns)

    return theta, log_probs_proposal


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

        # return self.likelihood.log_prob(theta) + self.prior.ln_prob(theta)


def plot_diagnostics(theta, outdir):
    # Plot weights
    weights = np.array(theta.pop("weights"))
    ESS = get_ESS(weights)
    log_probs_proposal = np.array(theta["log_probs_proposal"])
    log_probs_target = np.array(theta.pop("log_probs_target"))
    print(f"Number of samples:             {len(weights)}")
    print(f"Effective sample size (ESS):   {ESS:.0f} ({100 * ESS / len(weights):.2f}%)")

    plt.clf()
    y = weights / np.mean(weights)
    x = log_probs_proposal
    plt.xlabel("proposal log_prob")
    plt.ylabel("Weights (normalized to mean 1)")
    y_lower = 1e-4
    y_upper = math.ceil(
        np.max(y) / 10 ** math.ceil(np.log10(np.max(y)) - 1)
    ) * 10 ** math.ceil(np.log10(np.max(y)) - 1)
    plt.ylim(y_lower, y_upper)
    n_below = len(np.where(y < y_lower)[0])
    n_above = len(np.where(y > y_upper)[0])
    plt.yscale("log")
    plt.title(f"IS Weights. {n_below} below {y_lower}, {n_above} above {y_upper}")
    plt.scatter(x, y, s=0.5)
    plt.savefig(join(outdir, "weights.png"))

    plt.clf()
    x = log_probs_proposal - np.max(log_probs_proposal)
    y = log_probs_target - np.max(log_probs_target)
    plt.xlabel("Proposal log_prob (NDE)")
    plt.ylabel("Target log_prob (Likelihood x Prior)")
    y_lower, y_upper = -20, 0
    plt.ylim(y_lower, y_upper)
    n_below = len(np.where(y < y_lower)[0])
    plt.title(f"Target log_probs. {n_below} below {y_lower}.")
    plt.scatter(x, y, s=0.5)
    plt.savefig(join(outdir, "log_probs.png"))

    # cornerplot with unweighted vs. weighted samples
    weights = weights / np.mean(weights)
    threshold = 1e-3
    inds = np.where(weights > threshold)[0]
    theta_new = theta.loc[inds]
    weights_new = weights[inds]
    print(
        f"Generating cornerplot with {len(theta_new)} out of {len(theta)} IS samples."
    )

    c = ChainConsumer()
    c.add_chain(theta[: len(theta_new)], weights=None, color="orange", name="dingo")
    c.add_chain(theta_new, weights=weights_new, color="red", name="dingo-is")
    N = 2
    c.configure(
        linestyles=["-"] * N,
        linewidths=[1.5] * N,
        sigmas=[np.sqrt(2) * scipy.special.erfinv(x) for x in [0.5, 0.9]],
        shade=[False] + [True] * (N - 1),
        shade_alpha=0.3,
        bar_shade=False,
        label_font_size=10,
        tick_font_size=10,
        usetex=False,
        legend_kwargs={"fontsize": 30},
        kde=0.7,
    )
    c.plotter.plot(filename=join(outdir, "cornerplot_dingo-is.pdf"))


def get_ESS(weights):
    weights = np.array(weights)
    return np.sum(weights) ** 2 / np.sum(weights ** 2)


def main():
    # parse args, load settings, load dingo parameter samples
    args = parse_args()
    with open(args.settings, "r") as fp:
        settings = yaml.safe_load(fp)
    samples = pd.read_pickle(settings["nde"]["data"]["parameter_samples"])
    metadata = samples.attrs

    # Step 1: Build proposal distribution.
    #
    # We use the dingo posterior as our proposal distribution. We need to be able to
    # sample from, and evaluate this distribution. Here, we therefore train an
    # unconditional neural density estimator (nde) to recover the posterior density from
    # dingo samples. This typically required, since one loses the tractable likelihood
    # when using GNPE for dingo. This is not a big deal though, since we can cheaply
    # sample from the dingo posterior, such that one can easily train an unconditional
    # nde to recover the posterior density.

    if not "log_prob" in samples.columns:
        event_name = str(
            metadata["event"]["time_event"]
        )  # use gps time as name for now
        nde_name = join(args.outdir, f"nde-{event_name}.pt")
        if isfile(nde_name):
            print(f"Loading nde at {nde_name} for event {event_name}.")
            nde = PosteriorModel(
                nde_name,
                device=settings["nde"]["training"]["device"],
                load_training_info=False,
            )
        else:
            print(f"Training new nde for event {event_name}.")
            nde = train_unconditional_density_estimator(
                samples, settings["nde"], args.outdir
            )
            print(f"Renaming trained nde model to {nde_name}.")
            rename(join(args.outdir, "model_latest.pt"), nde_name)

    # Step 2: Build target distribution.
    #
    # Our target distribution is the posterior p(theta|d) = p(d|theta) * p(theta) / p(d).
    # For SIR, we need evaluate the *unnormalized* posterior, so we only need the
    # likelihood p(d|theta) and the prior p(theta), but not the evidence p(d).

    # build likelihood
    # metadata["model"]["dataset_settings"]["waveform_generator"]["approximant"] = "SEOBNRv4PHM"
    likelihood = build_stationary_gaussian_likelihood(
        # this should be set automatically from the samples
        metadata,
        args.event_dataset,
        settings.get("wfg_frequency_range", None),
    )
    # build prior
    intrinsic_prior = metadata["model"]["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = get_extrinsic_prior_dict(
        metadata["model"]["train_settings"]["data"]["extrinsic_prior"]
    )
    # merge priors, keep extrinsic prior if in conflict (e.g. for luminosity distance
    # which is in both, in extrinsic_prior, and as a reference value in intrinsic prior)
    prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})
    # wrap likelihood and prior to unnormalized posterior
    posterior = UnnormalizedPosteriorDensityBBH(likelihood, prior)

    # Step 3: SIR step
    #
    # Sample from proposal distribution q(theta|d) and reweight the samples theta_i with
    #
    #       w_i = p(theta_i|d) / q(theta_i|d)
    #
    # to obtain weighted samples from the proposal distribution.

    if "log_prob" in samples.columns:
        num_samples = len(samples)
        log_probs_proposal = np.array(samples["log_prob"])
        theta = samples.drop(columns="log_prob")
    else:
        num_samples = settings["num_samples"]
        # sample from proposal distribution, and get the log_prob densities
        print(f"Generating {num_samples} samples from proposal distribution.")
        theta, log_probs_proposal = get_samples_and_log_probs_from_proposal(
            nde, num_samples
        )

    # compute the unnormalized target posterior density for each sample
    t0 = time.time()
    print(f"Computing unnormalized target posterior density for {num_samples} samples.")
    log_probs_target = posterior.log_prob_multiprocessing(
        theta, settings.get("num_processes", 1)
    )
    print(f"Done. This took {time.time() - t0:.2f} seconds.")

    # compute weights, save weighted samples
    log_weights = log_probs_target - log_probs_proposal
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.mean(weights)
    theta.insert(theta.shape[1], "weights", weights)
    theta.insert(theta.shape[1], "log_probs_proposal", log_probs_proposal)
    theta.insert(theta.shape[1], "log_probs_target", log_probs_target)
    theta.to_pickle(join(args.outdir, "weighted_samples.pkl"))

    diagnositcs_dir = join(args.outdir, "IS-diagnostics")
    if not exists(diagnositcs_dir):
        makedirs(diagnositcs_dir)
    plot_diagnostics(theta, diagnositcs_dir)


if __name__ == "__main__":
    main()
