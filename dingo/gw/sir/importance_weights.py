"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""
from os import rename
from os.path import dirname, join, isfile
import numpy as np
import torch
import argparse

from dingo.core.models import PosteriorModel
from dingo.gw.likelihood import build_stationary_gaussian_likelihood


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

    def log_prob(self, theta):
        return self.likelihood.log_prob(theta) + self.prior.ln_prob(theta)


def main():
    from dingo.core.density import train_unconditional_density_estimator
    import yaml
    import pandas as pd
    from dingo.gw.gwutils import get_extrinsic_prior_dict
    from dingo.gw.prior import build_prior_with_defaults

    # parse args, load settings, load dingo parameter samples
    args = parse_args()
    with open(args.settings, "r") as fp:
        settings = yaml.safe_load(fp)
    samples = pd.read_pickle(settings["nde"]["data"]["parameter_samples"])
    # samples.attrs["event"] = {
    #     "time_event": 1126259462.4,
    #     "time_psd": 1024,
    #     "time_buffer": 2.0,
    # }
    # samples.to_pickle(settings["nde"]["data"]["parameter_samples"])
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

    event_name = str(metadata["event"]["time_event"])  # use gps time as name for now
    nde_name = join(args.outdir, f"nde-{event_name}.pt")
    if isfile(nde_name):
        print(f"Loading nde at {nde_name} for event {event_name}.")
        nde = PosteriorModel(
            nde_name,
            device=settings["nde"]["training"]["device"],
            load_training_info=False,
        )
    else:
        print(f"Training new for event {event_name}.")
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
    likelihood = build_stationary_gaussian_likelihood(samples.attrs, args.event_dataset)
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

    # sample from proposal distribution, and get the log_prob densities
    num_samples = settings["num_samples"]
    mean, std = nde.metadata["train_settings"]["data"]["standardization"].values()
    mean = np.array([v for v in mean.values()])
    std = np.array([v for v in std.values()])
    print(f"Generating {num_samples} samples from proposal distribution.")
    nde.model.eval()
    with torch.no_grad():
        theta = nde.model.sample(num_samples=num_samples)
        log_probs_proposal = nde.model.log_prob(theta).cpu().numpy()
    theta = theta.cpu().numpy() * std + mean
    theta = pd.DataFrame(theta, columns=samples.columns)
    #
    # import bilby
    # result = bilby.result.read_in_result(
    #     filename=join(args.outdir, "GW150914_result.json"))
    # theta_bilby = result.posterior[samples.columns]
    # theta_bilby["geocent_time"] -= likelihood.t_ref
    # l_bilby = result.log_likelihood_evaluations
    # # (np.mean(theta_bilby) - np.mean(theta)) / np.std(theta_bilby)
    #
    # theta = theta_bilby

    # compute the unnormalized target posterior density for each sample
    log_probs_target = []
    # likelihoods = []
    # priors = []
    print(f"Computing unnormalized target posterior density for {num_samples} samples.")
    from tqdm import tqdm
    for idx in tqdm(range(num_samples)):
        try:
            # priors.append(prior.ln_prob(dict(theta.iloc[idx])))
            # likelihoods.append(likelihood.log_prob(dict(theta.iloc[idx])))
            log_probs_target.append(posterior.log_prob(dict(theta.iloc[idx])))
        except:
            # likelihoods.append(-np.inf)
            # priors.append(-np.inf)
            log_probs_target.append(-np.inf)

    log_probs_target = np.array(log_probs_target)
    # likelihoods = np.array(likelihoods)
    # priors = np.array(priors)

    weights = log_probs_target - log_probs_proposal
    w = np.exp(weights - np.max(weights))
    w_norm = w / np.sum(w)
    print(np.sort(w_norm)[:20:-1])
    # test_samples = pd.DataFrame(samples[num_train_samples:], columns=parameters)
    theta.insert(theta.shape[1], "weights", w_norm)
    theta.to_pickle(join(args.outdir, "weighted_samples.pkl"))


    import matplotlib.pyplot as plt

    x = log_probs_proposal
    y = log_probs_target - np.max(log_probs_target)

    plt.clf()
    plt.ylabel("Likelihood x Prior log_prob")
    plt.xlabel("NDE log_prob")
    plt.ylim((-20, 0))
    plt.xlim((-15, 0))
    plt.scatter(x, y, s=0.5)
    plt.savefig(join(args.outdir, "scatter_plot.png"))
    plt.show()


    # cornerplot with reweighted samples
    import scipy
    from chainconsumer import ChainConsumer
    N = 2
    c = ChainConsumer()
    c.add_chain(theta, weights=None, color="orange", name="dingo")
    c.add_chain(theta, weights=w_norm, color="red", name="dingo-SIR")
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
    c.plotter.plot(filename=join(args.outdir, "cornerplot.pdf"))


    log_probs_target_ref = []
    likelihoods_ref = []
    priors_ref = []
    for idx in range(num_samples):
        try:
            priors_ref.append(prior.ln_prob(dict(samples.iloc[idx])))
            likelihoods_ref.append(likelihood.log_prob(dict(samples.iloc[idx])))
            log_probs_target_ref.append(posterior.log_prob(dict(samples.iloc[idx])))
        except:
            log_probs_target_ref.append(-np.inf)
    log_probs_target_ref = np.array(log_probs_target_ref)
    likelihoods_ref = np.array(likelihoods_ref)
    priors_ref = np.array(priors_ref)


    # plot weights
    plt.xlabel("log_weights")
    plt.hist(weights[~np.isinf(weights)] - np.max(weights), bins=100)
    plt.savefig(join(args.outdir, "weights.png"))
    plt.show()

    # plot nde density
    plt.xlabel("log_prob")
    plt.title("nde density")
    plt.hist(log_probs_proposal, bins=100)
    plt.savefig(join(args.outdir, "nde.png"))
    plt.show()

    # plot prior densities of original gnpe samples ("ref") and nde samples
    plt.xlabel("log_prob")
    plt.title("prior densities")
    plt.hist(priors_ref[~np.isinf(priors_ref)] - np.max(priors), bins=100, label="GNPE")
    plt.hist(
        priors[~np.isinf(priors)] - np.max(priors), bins=100, label="Unconditional NDE"
    )
    plt.legend()
    plt.savefig(join(args.outdir, "priors.png"))
    plt.show()

    # plot likelihoods densities of original gnpe samples ("ref") and nde samples
    plt.xlabel("log_prob")
    plt.title("likelihood densities")
    plt.xlim((-100, 0))
    plt.hist(
        likelihoods_ref[~np.isinf(likelihoods_ref)] - np.max(likelihoods),
        bins=1000,
        label="GNPE",
    )
    plt.hist(
        likelihoods[~np.isinf(likelihoods)] - np.max(likelihoods),
        bins=1000,
        alpha=0.8,
        label="Unconditional NDE",
    )
    plt.legend()
    plt.savefig(join(args.outdir, "likelihoods.png"))
    plt.show()

    # zoom out for likelihoods
    plt.yscale("log")
    plt.title("likelihood densities")
    plt.xlabel("log_prob")
    plt.hist(
        likelihoods_ref[~np.isinf(likelihoods_ref)] - np.max(likelihoods),
        bins=100,
        label="GNPE",
    )
    plt.legend()
    plt.savefig(join(args.outdir, "likelihoods-gnpe.png"))
    plt.show()
    plt.yscale("log")
    plt.xlabel("log_prob")
    plt.title("likelihood densities")
    plt.hist(
        likelihoods[~np.isinf(likelihoods)] - np.max(likelihoods),
        bins=100,
        label="Unconditional NDE",
    )
    plt.legend()
    plt.savefig(join(args.outdir, "likelihoods-nde.png"))
    plt.show()

    plt.hist(likelihoods - np.max(likelihoods), bins=100)
    plt.show()
    plt.hist(priors[~np.isinf(priors)] - np.max(priors), bins=100)
    plt.show()
    plt.hist(log_probs_proposal - np.max(log_probs_proposal), bins=100)
    plt.show()
    weights = log_probs_target - log_probs_proposal
    plt.hist(weights[~np.isinf(weights)] - np.max(weights), bins=100)
    plt.show()

    plt.hist(likelihoods_ref - np.max(likelihoods_ref), bins=100)
    plt.show()
    plt.hist(priors_ref[~np.isinf(priors_ref)] - np.max(priors_ref), bins=100)
    plt.show()
    plt.hist(
        log_probs_target_ref[~np.isinf(log_probs_target_ref)]
        - np.max(log_probs_target_ref, bins=100)
    )
    plt.show()

    f = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples"
        "/01_Pv2/model_latest_1.pt"
    )
    model = PosteriorModel(model_filename=f, device="cpu")
    print("done")


if __name__ == "__main__":
    main()
