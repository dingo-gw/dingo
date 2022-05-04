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
import argparse

from dingo.core.models import PosteriorModel
from dingo.gw.likelihood import build_stationary_gaussian_likelihood
from dingo.core.density import train_unconditional_density_estimator
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.posterior import UnnormalizedPosterior


def parse_args():
    parser = argparse.ArgumentParser(
        description="Importance sampling (IS) for dingo models."
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Path to settings file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for the unconditional nde and IS results.",
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
    # The standardization has an impact on the log_prob. For the computation of the
    # Bayesian evidence, the standardization of the proposal distribution (the nde) and
    # the target distribution (likelihood * prior) must be the same with respect to the
    # parameters theta. Since the prior is not standardized (it is a density in the
    # original parameter space), we simply undo the standardization of the nde to
    # restore compatibility with the prior. The contribution to the log_prob is thus
    # given by log(prod_i(std_i)) = sum_i log(std_i).
    log_probs_proposal += np.sum(np.log(std))

    # convert to pd.DataFrame
    columns = nde.metadata["train_settings"]["data"]["standardization"]["mean"].keys()
    theta = pd.DataFrame(theta, columns=columns)

    return theta, log_probs_proposal


def get_log_probs_from_proposal(nde, theta):
    """
    Compute the log_prob of, which is represented by an unconditional nde, from theta.

    Parameters
    ----------
    nde: dingo.core.models.PosteriorModel
        Unconditional nde used as proposal distribution.
    theta: pd.DataFrame
        Dataframe with theta samples for log_prob evaluation.

    Returns
    -------
    log_probs: numpy.ndarray
        Array with log_probs of the theta samples.
    """
    # standardization
    mean, std = nde.metadata["train_settings"]["data"]["standardization"].values()
    mean = np.array([v for v in mean.values()])
    std = np.array([v for v in std.values()])
    theta = (torch.from_numpy(np.array(theta)).to(nde.device).float() - mean) / std

    nde.model.eval()
    with torch.no_grad():
        log_probs_proposal = nde.model.log_prob(theta).cpu().numpy()

    # The standardization has an impact on the log_prob. For the computation of the
    # Bayesian evidence, the standardization of the proposal distribution (the nde) and
    # the target distribution (likelihood * prior) must be the same with respect to the
    # parameters theta. Since the prior is not standardized (it is a density in the
    # original parameter space), we simply undo the standardization of the nde to
    # restore compatibility with the prior. The contribution to the log_prob is thus
    # given by log(prod_i(std_i)) = sum_i log(std_i).
    log_probs_proposal += np.sum(np.log(std))

    return log_probs_proposal


def get_evidence(log_probs_target, log_probs_proposal):
    """
    Compute the Bayesian evidence p(d). The target distribution is the unnormalized
    posterior

            p^(theta) = p(d|theta) * p(theta)

    where p(d|theta) is the likelihood and p(theta) is the prior, and we leave the
    d-dependence implicit. For importance sampling we use the proposal distribution
    q(theta). We can express the evidence in terms of p^(theta) and q(theta) as

            p(d) = int d_theta p(d|theta) * p(theta)
                 = int d_theta p^(theta)
                 = int d_theta p^(theta) / q(theta) * q(theta).

    We evaluate this integral in the Monte Carlo approximation, by sampling
    theta_i ~ q(theta) and evaluating p^(theta_i) / q(theta_i),

            p(d) = 1/N * sum_{theta_i } p^(theta_i) / q(theta_i)
                 = 1/N * sum_{theta_i } w(theta_i),

    where w(theta_i) = p^(theta_i) / q(theta_i) are the importance weights of the
    samples. The log evidence is thus given by

            log(p(d)) = -log(N) + log sum exp(log_w),

    where the log weights are given by

            log_w = log(p^(theta)) - log(q(theta))
                  = log_probs_target - log_probs_proposal.

    Note: q(theta) and p(theta) need to be densities in the same parameter space, i.e.,
    their standardizations need to match.

    Parameters
    ----------
    log_probs_target: np.ndarray
        Array with log_probs of the samples from the proposal distribution, which is
        prior * likelihood.
    log_probs_proposal: np.ndarray
        Array with log_probs of the samples from the proposal distribution, which is
        represented by the unconditional nde.

    Returns
    -------
    log_evidence: float
        Log evidence log(p(d)).
    """
    log_w = log_probs_target - log_probs_proposal
    N = len(log_w)
    # Use the logsumexp trick to avoid numerical underflow.
    alpha = np.max(log_w)
    log_evidence = -np.log(N) + np.log(np.sum(np.exp(log_w - alpha))) + alpha
    return log_evidence


def plot_posterior_slice(
    posterior,
    nde,
    theta,
    theta_range,
    log_evidence,
    outname=None,
    num_processes=1,
    n_grid=200,
):
    # repeat theta n_grid times
    theta_grid = pd.DataFrame(
        np.repeat(np.array(theta)[np.newaxis], n_grid, axis=0),
        columns=theta.keys(),
    )
    plt.clf()
    fig, ax = plt.subplots(3, 5, figsize=(30, 12))
    for idx, param in enumerate(theta.keys()):
        # axis with scan for param
        param_axis = np.linspace(theta_range[param][0], theta_range[param][1], n_grid)
        # build theta_grid for param
        theta_param = theta_grid.copy()
        theta_param[param] = param_axis
        # evaluate the posterior at theta_grid
        log_probs_target = (
            posterior.log_prob_multiprocessing(theta_param, num_processes)
            - log_evidence
        )
        # evaluate nde at theta_grid
        log_probs_proposal = get_log_probs_from_proposal(nde, theta_param)

        # plot
        i, j = idx // 5, idx % 5
        ax[i, j].set_xlabel(param)
        ax[i, j].axvline([theta[param]], color="black", label="theta")
        ax[i, j].plot(param_axis, np.exp(log_probs_target), label="target")
        ax[i, j].plot(param_axis, np.exp(log_probs_proposal), label="proposal")

    plt.legend()
    if outname is not None:
        plt.savefig(outname)
    else:
        plt.show()


def plot_diagnostics(
    theta,
    outdir,
    theta_slice_plots=None,
    posterior=None,
    nde=None,
    num_processes=1,
    n_grid=200,
):
    weights = np.array(theta.pop("weights"))
    # compute ESS
    ESS = get_ESS(weights)
    log_probs_proposal = np.array(theta["log_probs_proposal"])
    log_probs_target = np.array(theta.pop("log_probs_target"))
    print(f"Number of samples:             {len(weights)}")
    print(f"Effective sample size (ESS):   {ESS:.0f} ({100 * ESS / len(weights):.2f}%)")
    # Compute log_evidence
    log_evidence = get_evidence(log_probs_target, log_probs_proposal)
    print(f"Log evidence:                  {log_evidence:.2f}")
    # Normalize target log_probs
    log_probs_target = log_probs_target - log_evidence

    # Plot weights
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
    plt.yscale("log")
    plt.title(
        f"IS Weights. {n_below} below {y_lower}. "
        f"ESS {ESS:.0f} ({100 * ESS / len(weights):.2f}%)."
    )
    plt.scatter(x, y, s=0.5)
    plt.savefig(join(outdir, "weights.png"))

    plt.clf()
    x = log_probs_proposal
    y = log_probs_target
    plt.xlabel("NDE log_prob (proposal)")
    plt.ylabel("Posterior log_prob (target)")
    y_lower, y_upper = np.max(y) - 20, np.max(y)
    plt.ylim(y_lower, y_upper)
    n_below = len(np.where(y < y_lower)[0])
    plt.title(
        f"Target log_probs. {n_below} below {y_lower:.2f}. "
        f"Log_evidence {log_evidence:.2f}."
    )
    plt.scatter(x, y, s=0.5)
    plt.plot([y_upper - 20, y_upper], [y_upper - 20, y_upper], color="black")
    plt.savefig(join(outdir, "log_probs.png"))

    if theta_slice_plots is not None:
        if posterior is None or nde is None:
            raise ValueError("Must provide posterior and nde.")
        # global range for parameter scan
        theta_range = {
            k: (np.min(theta[k]), np.max(theta[k])) for k in theta_slice_plots.columns
        }
        # generate slice plots for each theta sample
        for idx, (_, theta_idx) in enumerate(theta_slice_plots.iterrows()):
            plot_posterior_slice(
                posterior,
                nde,
                theta_idx,
                theta_range,
                log_evidence,
                num_processes=num_processes,
                outname=join(outdir, f"theta_{idx}_posterior_slice.pdf"),
                n_grid=n_grid,
            )

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
    # for time marginalization, we drop geocent time from the samples
    if "time_marginalization" in settings and "geocent_time" in samples:
        samples.drop("geocent_time", axis=1, inplace=True)

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
        nde_name = settings["nde"].get(
            "path", join(args.outdir, f"nde-{event_name}.pt")
        )
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
    else:
        nde = None

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
        settings.get("event_dataset", None),
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
    # posterior = UnnormalizedPosterior(likelihood, prior)
    posterior = UnnormalizedPosterior(
        likelihood, prior, settings.get("time_marginalization", None)
    )

    # Step 3: SIR step
    #
    # Sample from proposal distribution q(theta|d) and reweight the samples theta_i with
    #
    #       w_i = p(theta_i|d) / q(theta_i|d)
    #
    # to obtain weighted samples from the proposal distribution.

    if "log_prob" in samples.columns:
        num_samples = settings.get("num_samples", len(samples))
        theta = samples.sample(num_samples)
        log_probs_proposal = np.array(theta["log_prob"])
        theta = theta.drop(columns="log_prob")
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

    # compute weights, save weighted samples with metadata
    log_weights = log_probs_target - log_probs_proposal
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.mean(weights)
    theta.insert(theta.shape[1], "weights", weights)
    theta.insert(theta.shape[1], "log_probs_proposal", log_probs_proposal)
    theta.insert(theta.shape[1], "log_probs_target", log_probs_target)
    theta.attrs = {"dingo_settings": metadata, "is_settings": settings}
    theta.to_pickle(join(args.outdir, "weighted_samples.pkl"))

    diagnostics_dir = join(args.outdir, "IS-diagnostics")
    if not exists(diagnostics_dir):
        makedirs(diagnostics_dir)
    if settings.get("n_slice_plots", 0) > 0:
        theta_slice_plots = theta.sample(settings["n_slice_plots"]).drop(
            columns=["weights", "log_probs_proposal", "log_probs_target"]
        )
    else:
        theta_slice_plots = None
    # theta_slice_plots = theta.iloc[[np.argmax(theta["log_probs_target"])]].drop(
    #     columns=["weights", "log_probs_proposal", "log_probs_target"]
    # )
    plot_diagnostics(
        theta,
        diagnostics_dir,
        theta_slice_plots=theta_slice_plots,
        posterior=posterior,
        nde=nde,
        num_processes=settings.get("num_processes", 1),
    )


if __name__ == "__main__":
    main()
