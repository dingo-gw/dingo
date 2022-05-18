"""
Step 1: Train unconditional nde
Step 2: Set up likelihood and prior
"""
import yaml
from os import rename, makedirs
from os.path import dirname, join, isfile, exists
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy
import argparse

from dingo.core.models import PosteriorModel
from dingo.core.samplers import Sampler
from dingo.core.samples_dataset import SamplesDataset
from dingo.gw.inference.gw_samplers import GWSamplerUnconditional
from dingo.core.density import train_unconditional_density_estimator


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


def plot_posterior_slice2d(
    sampler, theta, theta_range, n_grid=100, num_processes=1, outname=None
):
    # prepare theta grid
    keys = list(theta_range.keys())
    axis0 = np.linspace(theta_range[keys[0]][0], theta_range[keys[0]][1], n_grid)
    axis1 = np.linspace(theta_range[keys[1]][0], theta_range[keys[1]][1], n_grid)
    full_axis0, full_axis1 = np.meshgrid(axis0, axis1)

    p0, p1 = theta[keys[0]], theta[keys[1]]
    theta_grid = pd.DataFrame(
        np.repeat(np.array(theta)[np.newaxis], n_grid ** 2, axis=0),
        columns=theta.keys(),
    )
    theta_grid[keys[0]] = full_axis0.flatten()
    theta_grid[keys[1]] = full_axis1.flatten()

    # compute log_prob
    log_probs_target = (
        sampler.likelihood.log_likelihood_multi(theta_grid, num_processes)
        # + sampler.prior.ln_prob(theta_grid, axis=0)
        - sampler.log_evidence
    )
    log_probs_target -= np.max(log_probs_target)
    plt.clf()
    plt.imshow(np.exp(log_probs_target).reshape((n_grid, n_grid)), cmap="viridis")
    plt.colorbar()
    ticks_positions = np.arange(0, n_grid, n_grid // 5)
    plt.xticks(ticks_positions, labels=[f"{v:.2f}" for v in axis0[ticks_positions]])
    plt.xlabel(keys[0])
    plt.yticks(ticks_positions, labels=[f"{v:.2f}" for v in axis1[ticks_positions]])
    plt.ylabel(keys[1])

    if outname is not None:
        plt.savefig(outname)
    else:
        plt.show()


def plot_posterior_slice(
    sampler,
    theta,
    theta_range,
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
            sampler.likelihood.log_likelihood_multi(theta_param, num_processes)
            + sampler.prior.ln_prob(theta_param, axis=0)
            - sampler.log_evidence
        )
        # evaluate nde at theta_grid
        # log_probs_proposal = get_log_probs_from_proposal(nde, theta_param)
        log_probs_proposal = sampler.log_prob(theta_param)

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
    sampler: Sampler,
    outdir,
    num_processes=1,
    num_slice_plots=0,
    n_grid_slice1d=200,
    n_grid_slice2d=100,
    params_slice2d=None,
):
    theta = sampler.samples
    weights = theta["weights"].to_numpy()
    log_probs_proposal = theta["log_prob"].to_numpy()

    log_prior = theta["log_prior"].to_numpy()
    log_likelihood = theta["log_likelihood"].to_numpy()
    log_probs_target = log_prior + log_likelihood

    ESS = sampler.effective_sample_size
    log_evidence = sampler.log_evidence

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

    # plot log probs
    plt.clf()
    x = log_probs_proposal
    y = log_probs_target - log_evidence
    plt.xlabel("NDE log_prob (proposal)")
    plt.ylabel("Posterior log_prob (target) - log_evidence")
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

    # plot slice plots
    if num_slice_plots > 0:
        theta_slice_plots = theta.sample(num_slice_plots).drop(
            columns=["weights", "log_prob", "log_prior", "log_likelihood"]
        )
        # global range for 1D parameter scan
        theta_range_1d = {
            k: (np.min(theta[k]), np.max(theta[k])) for k in theta_slice_plots.columns
        }
        # generate slice plots for each theta sample
        for idx, (_, theta_idx) in enumerate(theta_slice_plots.iterrows()):
            # 1d slice plots
            plot_posterior_slice(
                sampler,
                theta_idx,
                theta_range_1d,
                num_processes=num_processes,
                n_grid=n_grid_slice1d,
                outname=join(outdir, f"theta_{idx}_posterior_slice1d.pdf"),
            )
            # optionally, plot 2d slice plots
            if params_slice2d is not None:
                # Get parameter ranges for 2d scan.
                # We set this as a 1 std area around the respective parameter values,
                # except for the phase which we scan in [0, 2pi].
                stds = {k: np.std(theta[k]) for k in theta.keys()}
                theta_range_2d = {
                    k: (theta_idx[k] - stds[k], theta_idx[k] + stds[k])
                    for k in theta_idx.keys()
                }
                theta_range_2d["phase"] = (0, 2 * np.pi)
                for param_pair in params_slice2d:
                    plot_posterior_slice2d(
                        sampler,
                        theta_idx,
                        {k: theta_range_2d[k] for k in param_pair},
                        num_processes=num_processes,
                        n_grid=n_grid_slice2d,
                        outname=join(
                            outdir,
                            f"theta_{idx}_posterior_slice2d_"
                            f"{param_pair[0]}-{param_pair[1]}.pdf",
                        ),
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


def main():
    # parse args, load settings, load dingo parameter samples
    args = parse_args()
    with open(args.settings, "r") as fp:
        settings = yaml.safe_load(fp)
    samples_dataset = SamplesDataset(
        file_name=settings["nde"]["data"]["parameter_samples"]
    )
    metadata = samples_dataset.settings
    samples = samples_dataset.samples
    # for time marginalization, we drop geocent time from the samples
    inference_parameters = metadata["train_settings"]["data"][
        "inference_parameters"
    ].copy()
    time_marginalization_kwargs = settings.get("time_marginalization", None)
    if time_marginalization_kwargs is not None and "geocent_time" in samples:
        samples.drop("geocent_time", axis=1, inplace=True)
        inference_parameters.remove("geocent_time")
    settings["nde"]["data"]["inference_parameters"] = inference_parameters

    # Step 1: Build proposal distribution.
    #
    # We use the dingo posterior as our proposal distribution. We need to be able to
    # sample from, and evaluate this distribution. Here, we therefore train an
    # unconditional neural density estimator (nde) to recover the posterior density from
    # dingo samples. This typically required, since one loses the tractable likelihood
    # when using GNPE for dingo. This is not a big deal though, since we can cheaply
    # sample from the dingo posterior, such that one can easily train an unconditional
    # nde to recover the posterior density.

    if "log_prob" not in samples.columns:
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
                samples_dataset, settings["nde"], args.outdir
            )
            print(f"Renaming trained nde model to {nde_name}.")
            rename(join(args.outdir, "model_latest.pt"), nde_name)
    else:
        raise NotImplementedError(
            "Cannot currently perform importance sampling based "
            "on just a samples dataset, even with log_prob "
            "included. Please start with a posterior model."
        )

    # Step 2: Sample from proposal.

    nde_sampler = GWSamplerUnconditional(model=nde)
    print(f'Generating {settings["num_samples"]} samples from proposal distribution.')
    nde_sampler.run_sampler(num_samples=settings["num_samples"])

    # Step 3: Importance sample.
    #
    # Our target distribution is the posterior p(theta|d) = p(d|theta) * p(theta) / p(
    # d). For importance sampling, we need to evaluate the *unnormalized* posterior,
    # so we only need the likelihood p(d|theta) and the prior p(theta), but not the
    # evidence p(d).
    #
    # Sample from proposal distribution q(theta|d) and reweight the samples theta_i with
    #
    #       w_i = p(theta_i|d) / q(theta_i|d)
    #
    # to obtain weighted samples from the proposal distribution.

    print(f"Importance sampling.")
    nde_sampler.importance_sample(
        num_processes=settings.get("num_processes", 1),
        time_marginalization_kwargs=time_marginalization_kwargs,
    )
    nde_sampler.print_summary()
    nde_sampler.to_hdf5(label="weighted", outdir=args.outdir)
    samples = nde_sampler.samples

    # Diagnostics
    diagnostics_dir = join(args.outdir, "IS-diagnostics")
    if not exists(diagnostics_dir):
        makedirs(diagnostics_dir)
    print("Plotting diagnostics.")
    plot_diagnostics(
        nde_sampler,
        diagnostics_dir,
        num_processes=settings.get("num_processes", 1),
        **settings.get("slice_plots", {}),
    )


if __name__ == "__main__":
    main()
