from os.path import join
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from dingo.core.utils.plotting import plot_corner_multi
from dingo.gw.result import Result


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
        np.repeat(np.array(theta)[np.newaxis], n_grid**2, axis=0),
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
    parameters=None,
    normalize_conditionals=False,
):
    # Put a cap on number of processes to avoid overhead.
    # num_processes = min(num_processes, n_grid // 50)

    # repeat theta n_grid times
    theta_grid = pd.DataFrame(
        np.repeat(np.array(theta)[np.newaxis], n_grid, axis=0),
        columns=theta.keys(),
    )
    plt.clf()
    if parameters is None:
        parameters = theta.keys()
    # rows, columns = 3, 5
    columns = math.ceil(np.sqrt(len(parameters)))
    rows = math.ceil(len(parameters) / columns)
    fig, ax = plt.subplots(
        rows, columns, figsize=(columns * 6, rows * 4), squeeze=False
    )
    for idx, param in enumerate(parameters):
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
        target = np.exp(log_probs_target)
        proposal = np.exp(log_probs_proposal)
        if normalize_conditionals:
            target /= target.sum()
            proposal /= proposal.sum()
        i, j = idx // columns, idx % columns
        ax[i, j].set_xlabel(param)
        ax[i, j].axvline([theta[param]], color="black", label="theta")
        ax[i, j].plot(param_axis, target, label="target")
        ax[i, j].plot(param_axis, proposal, label="proposal")

    plt.legend()
    if outname is not None:
        plt.savefig(outname)
    else:
        plt.show()


def plot_diagnostics(
    result: Result,
    outdir,
    num_processes=1,
    num_slice_plots=0,
    n_grid_slice1d=200,
    n_grid_slice2d=100,
    params_slice2d=None,
):
    theta = result.samples
    theta = theta.drop(columns="delta_log_prob_target", errors="ignore")
    weights = theta["weights"].to_numpy()
    log_probs_proposal = theta["log_prob"].to_numpy()

    log_prior = theta["log_prior"].to_numpy()
    log_likelihood = theta["log_likelihood"].to_numpy()
    log_probs_target = log_prior + log_likelihood

    n_eff = result.n_eff
    sample_efficiency = result.sample_efficiency
    log_evidence = result.log_evidence
    log_evidence_std = result.log_evidence_std

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
        f"Effective samples: {n_eff:.0f} (Efficiency = {100 * sample_efficiency:.2f}%)."
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
        f"Log_evidence {log_evidence:.3f} +- {log_evidence_std:.3f}."
    )
    plt.scatter(x, y, s=0.5)
    plt.plot([y_upper - 20, y_upper], [y_upper - 20, y_upper], color="black")
    plt.savefig(join(outdir, "log_probs.png"))

    # # plot slice plots
    # if num_slice_plots > 0:
    #     theta_slice_plots = theta.sample(num_slice_plots).drop(
    #         columns=["weights", "log_prob", "log_prior", "log_likelihood"]
    #     )
    #     # global range for 1D parameter scan
    #     theta_range_1d = {
    #         k: (np.min(theta[k]), np.max(theta[k])) for k in theta_slice_plots.columns
    #     }
    #     # generate slice plots for each theta sample
    #     for idx, (_, theta_idx) in enumerate(theta_slice_plots.iterrows()):
    #         # 1d slice plots
    #         plot_posterior_slice(
    #             sampler,
    #             theta_idx,
    #             theta_range_1d,
    #             num_processes=num_processes,
    #             n_grid=n_grid_slice1d,
    #             outname=join(outdir, f"theta_{idx}_posterior_slice1d.pdf"),
    #         )
    #         # optionally, plot 2d slice plots
    #         if params_slice2d is not None:
    #             # Get parameter ranges for 2d scan.
    #             # We set this as a 1 std area around the respective parameter values,
    #             # except for the phase which we scan in [0, 2pi].
    #             stds = {k: np.std(theta[k]) for k in theta.keys()}
    #             theta_range_2d = {
    #                 k: (theta_idx[k] - stds[k], theta_idx[k] + stds[k])
    #                 for k in theta_idx.keys()
    #             }
    #             theta_range_2d["phase"] = (0, 2 * np.pi)
    #             for param_pair in params_slice2d:
    #                 plot_posterior_slice2d(
    #                     sampler,
    #                     theta_idx,
    #                     {k: theta_range_2d[k] for k in param_pair},
    #                     num_processes=num_processes,
    #                     n_grid=n_grid_slice2d,
    #                     outname=join(
    #                         outdir,
    #                         f"theta_{idx}_posterior_slice2d_"
    #                         f"{param_pair[0]}-{param_pair[1]}.pdf",
    #                     ),
    #                 )

    # cornerplot with unweighted vs. weighted samples
    weights = weights / np.mean(weights)
    threshold = 1e-3
    inds = np.where(weights > threshold)[0]
    theta_new = theta.loc[inds]
    weights_new = weights[inds]
    print(
        f"Generating cornerplot with {len(theta_new)} out of {len(theta)} IS samples."
    )

    plot_corner_multi(
        [theta[: len(theta_new)], theta_new],
        weights=[None, weights_new],
        labels=["Dingo", "Dingo-IS"],
        filename=join(outdir, "cornerplot_dingo-is.pdf"),
    )
