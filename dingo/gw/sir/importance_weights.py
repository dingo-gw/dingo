"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""
from os import rename
from os.path import dirname, join, isfile
import numpy as np
import torch
import pandas as pd
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
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
            log_likelihood = self.likelihood.log_prob(theta)
            log_prior = self.prior.ln_prob(theta)
            return log_likelihood + log_prior
        except:
            return -np.inf

        # return self.likelihood.log_prob(theta) + self.prior.ln_prob(theta)


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
        print(f"Training new nde for event {event_name}.")
        nde = train_unconditional_density_estimator(
            samples, settings["nde"], args.outdir
        )
        print(f"Renaming trained nde model to {nde_name}.")
        rename(join(args.outdir, "model_latest.pt"), nde_name)

    # diagnostics
    # from dingo.gw.inference.visualization import generate_cornerplot
    # samples_nde, _ = get_samples_and_log_probs_from_proposal(nde, 10000)
    # generate_cornerplot(
    #     {"name": "gnpe", "color": "orange", "samples": samples[:10000]},
    #     {"name": "nde", "color": "black", "samples": samples_nde},
    #     filename=join(args.outdir, "cornerplot-nde.png"),
    # )

    # Step 2: Build target distribution.
    #
    # Our target distribution is the posterior p(theta|d) = p(d|theta) * p(theta) / p(d).
    # For SIR, we need evaluate the *unnormalized* posterior, so we only need the
    # likelihood p(d|theta) and the prior p(theta), but not the evidence p(d).

    # build likelihood
    likelihood = build_stationary_gaussian_likelihood(metadata, args.event_dataset)
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

    theta_d_Pv2 = pd.read_pickle("/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples/04_Pv2/merged_dingo_samples_gps-1126259462.4_.pkl")

    # p_dPv2 = posterior.log_prob_multiprocessing(theta_d_Pv2[:1000], num_processes=8)
    # p_nde = posterior.log_prob_multiprocessing(samples[:1000], num_processes=8)


    # diagnostics
    from dingo.gw.inference.visualization import generate_cornerplot, load_ref_samples
    theta_LI = load_ref_samples("/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/02_gwpe/train_dir_max/cluster_models/GW150914_Pv2_LI.npz")
    theta_PRL = load_ref_samples(join("/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples/04_Pv2/GW150914_PRL.npz"),
                                 drop_geocent_time=False)

    import bilby
    result = bilby.result.read_in_result(
     filename=join("/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets"
                   "/dingo_samples/04_Pv2/GW150914_result_UDP.json"))
    theta_bilby = result.posterior[samples.columns]
    theta_bilby["geocent_time"] -= likelihood.t_ref
    theta_bilby = theta_bilby.sample(frac=1) # shuffle bilby data
    samples.attrs = {}
    posterior.log_prob(dict(samples.iloc[0]))
    posterior.log_prob(dict(theta_bilby.iloc[0]))

    l_bilby = result.log_likelihood_evaluations
    # p_nde = posterior.log_prob_multiprocessing(samples[:50_000], num_processes=8)
    p_bilby = posterior.log_prob_multiprocessing(theta_bilby[:50_000], num_processes=8)
    # p_PRL = posterior.log_prob_multiprocessing(theta_PRL[:1000], num_processes=8)
    # generate_cornerplot(
    #     {"name": "bilby", "color": "black", "samples": theta_bilby[:10_000]},
    #     {"name": "LI", "color": "blue", "samples": theta_LI[:10_000]},
    #     {"name": "gnpe", "color": "orange", "samples": samples[:10_000]},
    #     filename=join(args.outdir, "cornerplot-bilby-LI.pdf"),
    # )


    # Step 3: SIR step
    #
    # Sample from proposal distribution q(theta|d) and reweight the samples theta_i with
    #
    #       w_i = p(theta_i|d) / q(theta_i|d)
    #
    # to obtain weighted samples from the proposal distribution.

    num_samples = settings["num_samples"]
    # sample from proposal distribution, and get the log_prob densities
    print(f"Generating {num_samples} samples from proposal distribution.")
    theta, log_probs_proposal = get_samples_and_log_probs_from_proposal(nde, num_samples)
    # compute the unnormalized target posterior density for each sample
    import time
    t0 = time.time()
    print(f"Computing unnormalized target posterior density for {num_samples} samples.")
    log_probs_target = posterior.log_prob_multiprocessing(
        theta, settings.get("num_processes", 1)
    )
    print(f"Done. This took {time.time() - t0:.2f} seconds.")

    # weights = log_probs_target - log_probs_proposal
    # w = np.exp(weights - np.max(weights))
    # w_norm = w / np.sum(w)
    # print(np.sort(w_norm)[:20:-1])
    log_weights = log_probs_target - log_probs_proposal
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.mean(weights)

    threshold = 1e-3
    inds = np.where(weights > threshold)[0]
    theta_new = theta.loc[inds]
    weights_new = weights[inds]

    from chainconsumer import ChainConsumer
    import scipy
    from dingo.gw.inference.visualization import load_ref_samples
    theta_LI = load_ref_samples(
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel/tutorials/02_gwpe/train_dir_max/cluster_models/GW150914_Pv2_LI.npz")
    theta_LI.drop(columns="mass_ratio", inplace=True)
    theta_LI.drop(columns="chirp_mass", inplace=True)

    c = ChainConsumer()
    c.add_chain(theta_LI, color="blue", name="LI")
    c.add_chain(theta[:20_000], weights=None, color="orange", name="dingo")
    c.add_chain(theta_new, weights=weights_new, color="red", name="dingo-sir")
    N = 3
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
    c.plotter.plot(filename=join(args.outdir, "cornerplot_sir-300k-LI.pdf"))


    inds_low = list(set(np.where(weights > 1e-15)[0]) & set(np.where(weights < 1e-10)[0]))
    weights_low = weights[inds_low]
    theta_low = theta.loc[inds_low]

    c = ChainConsumer()
    c.add_chain(theta_LI, color="blue", name="LI")
    c.add_chain(theta_low, weights=None, color="orange", name="dingo low weights")
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
    c.plotter.plot(filename=join(args.outdir, "cornerplot_dingo-low-weights.pdf"))


    l_bilby = result.log_likelihood_evaluations
    p_bilby = posterior.log_prob_multiprocessing(theta_bilby, num_processes=8)
    priors_bilby = []
    for idx in range(len(theta_bilby)):
        try:
            priors_bilby.append(prior.ln_prob(dict(theta_bilby.iloc[idx])))
        except:
            priors_bilby.append(-np.inf)
    priors_bilby = np.array(priors_bilby)
    np.percentile(p_bilby - l_bilby - priors_bilby, 10)

    plt.title("Densities of bilby samples")
    plt.xlabel("Proposal density (NDE)")
    plt.ylabel(f"Target density (likelihood * prior) [{np.max(p_bilby):.2f}]")
    plt.scatter(log_probs_bilby, p_bilby - np.max(p_bilby), s=0.1)
    plt.savefig(join(args.outdir, "nde_densities_bilby-2.png"))
    plt.show()
    plt.clf()

    # test_samples = pd.DataFrame(samples[num_train_samples:], columns=parameters)
    theta.insert(theta.shape[1], "weights", weights)
    theta.insert(theta.shape[1], "log_probs_proposal", log_probs_proposal)
    theta.insert(theta.shape[1], "log_probs_target", log_probs_target)
    theta.to_pickle(join(args.outdir, "weighted_samples.pkl"))

    nde.model.eval()
    # standardize
    mean, std = nde.metadata["train_settings"]["data"]["standardization"].values()
    mean = np.array([v for v in mean.values()])
    std = np.array([v for v in std.values()])
    theta_bilby_torch = torch.tensor((np.array(theta_bilby) - mean) / std).float()
    with torch.no_grad():
        log_probs_bilby = nde.model.log_prob(theta_bilby_torch).cpu().numpy()

    import matplotlib.pyplot as plt
    plt.clf()
    plt.xlabel("nde density")
    plt.hist(log_probs_proposal, bins=100, density=True, label="nde samples")
    plt.hist(log_probs_bilby, bins=100, density=True, label="bilby samples")
    plt.show()



    import matplotlib.pyplot as plt
    plt.yscale("log")
    plt.ylim(1e-4, 1e3)
    plt.xlabel("proposal log_prob")
    plt.ylabel("Weights (normalized to mean 1)")
    plt.scatter(log_probs_proposal, weights, s=0.5)
    plt.savefig(join(args.outdir, "weights_scatter_plot.png"))
    plt.show()
    plt.clf()

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
