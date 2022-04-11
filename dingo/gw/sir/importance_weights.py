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

    # compute the unnormalized target posterior density for each sample
    log_probs_target = []
    print(f"Computing unnormalized target posterior density for {num_samples} samples.")
    for idx in range(num_samples):
        try:
            log_probs_target.append(posterior.log_prob(dict(theta.iloc[idx])))
        except:
            log_probs_target.append(-np.inf)
    log_probs_target = np.array(log_probs_target)

    weights = log_probs_target - log_probs_proposal
    w = weights - np.max(weights)
    # test_samples = pd.DataFrame(samples[num_train_samples:], columns=parameters)







    f = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples"
        "/01_Pv2/model_latest_1.pt"
    )
    model = PosteriorModel(model_filename=f, device="cpu")
    print("done")


if __name__ == "__main__":
    main()
