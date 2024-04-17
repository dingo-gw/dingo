import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import argparse
import textwrap
import os


def weighted_percentile_of_score(samples, value, weights=None):
    """
    Compute the percentile rank of a score relative to a list of scores
    with attached weights.
    
    Parameters
    ----------
    samples: array-like
        Array to which `value` is compared to
    value: float 
        A given score to compute the percentile for
    weights: array-like; optional
        Array of weights
    
    Returns
    -------
    percentile: float
    """
    if weights is None:
        weights = np.ones_like(samples)
    else:
        if len(weights) != len(samples):
            raise ValueError('weights and samples need to have the same length!')

    # Sort the samples and weights
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Calculate the cumulative sum of the weights and normalize to get the CDF
    cdf = np.cumsum(sorted_weights) / sorted_weights.sum()

    # Find the index of the first element in the sorted samples array
    # that is greater than or equal to the given value
    index = np.searchsorted(sorted_samples, value)

    # Look up the corresponding CDF value at that index to get the percentile
    percentile = cdf[min(index, len(cdf) - 1)] * 100
    return percentile


def make_pp_plot(
        percentiles, parameter_labels, filename, ks=True,
        confidence_interval=[0.68, 0.95, 0.997], confidence_interval_alpha=0.1,
        plot_projection=True, lw=1.0,
    ):
    """
    Make a probability-probability plot with confidence bands.

    Parameters
    ----------
    percentiles: array-like
        Array of percentiles of shape (n_posteriors, n_parameters)
    parameter_labels: array-like
        List of parameter labels
    filename: str
        Name of file to save the generated plot to.
    ks: bool
        Whether to display p-values from KS-tests in the legend.
    confidence_interval: list
        If not None show binomial confidence bands for the specified values.
    confidence_interval_alpha:
        Opacity for confidence bands.

    This function is based on code from
    https://github.com/stephengreen/lfi-gw/blob/master/notebooks/nde_evaluation_GW150914.ipynb
    and https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py#L2068            
    """
    percentiles = percentiles/100.
    nparams = percentiles.shape[-1]
    nposteriors = percentiles.shape[0]
    print('# posteriors:', nposteriors)

    ordered = np.sort(percentiles, axis=0)
    ordered = np.concatenate((np.zeros((1, nparams)), ordered, np.ones((1, nparams))))
    y = np.linspace(0, 1, nposteriors + 2)
        
    if plot_projection:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10,15), gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(10,10))
    
    for n in range(nparams):
        if ks:
            pvalue = stats.kstest(percentiles[:, n], 'uniform')[1]
            ax1.step(ordered[:, n], y, where='post', lw=lw, label=parameter_labels[n] + r' ({:.3g})'.format(pvalue))
        else:
            ax1.step(ordered[:, n], y, where='post', lw=lw, label=parameter_labels[n])

        if plot_projection:
            ax2.step(ordered[:, n], y - ordered[:, n], where='post', lw=lw, label=parameter_labels[n])

    if confidence_interval is not None:
        if isinstance(confidence_interval, float):
            confidence_interval = [confidence_interval]
        if isinstance(confidence_interval_alpha, float):
            confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)
        elif len(confidence_interval_alpha) != len(confidence_interval):
            raise ValueError(
                "confidence_interval_alpha must have the same length as confidence_interval")
    
        y = np.linspace(0, 1, 1001)
        for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
            edge_of_bound = (1. - ci) / 2.
            lower = stats.binom.ppf(1 - edge_of_bound, nposteriors, y) / nposteriors
            upper = stats.binom.ppf(edge_of_bound, nposteriors, y) / nposteriors
            # The binomial point percent function doesn't always return 0 @ 0,
            # so set those bounds explicitly to be sure.
            # Similarly set the values at 1 to 1.
            lower[0] = 0
            upper[0] = 0
            lower[-1] = 1
            upper[-1] = 1
            ax1.fill_between(y, lower, upper, alpha=alpha, color='k')
            if plot_projection:
                ax2.fill_between(y, lower - y, upper - y, alpha=alpha, color='k')
    
    ax1.plot(y, y, 'k--')
    ax1.legend(title=f"PP plot with {nposteriors} samples")
    ax1.set_xlabel(r'$p$')
    ax1.set_ylabel(r'$CDF(p)$')
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax1.set_aspect('equal', anchor='SW')

    if plot_projection:
        # ax2.set_aspect(0.5, anchor='SW')
        ax2.plot(y, y * 0, 'k--')
        ax2.set_xlabel(r'$p$')
        ax2.set_ylabel(r'$CDF(p) - p$')
        ax2.set_xlim((0, 1))
        ax2.set_ylim((-0.01, 0.01))


    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_loghist(x, bins):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(x, bins=logbins)
  plt.xscale('log')


if __name__ == "__main__":
    # analysis_dir = "/fast/groups/dingo/03_binary_neutron_stars/03_models/03_30M_datasets/02_GW190425_lowSpin/02_dataset-g_nn-xl_epochs-400/inference/02_pp-plots/01/"
    outdir = "/fast/groups/dingo/03_binary_neutron_stars/03_models/03_30M_datasets/02_GW190425_lowSpin/02_dataset-g_nn-xl_epochs-400/inference/02_pp-plots/03_hyperprior_100k/"
    datadir = os.path.join(outdir, "data")
    outdir = "."
    filenames = [f for f in os.listdir(datadir) if f.endswith(".pd")]
    dfs = [pd.read_pickle(os.path.join(datadir, f)) for f in filenames]
    data = pd.concat(dfs, ignore_index=True)

    percentiles = {k[len("dingo:"):]: np.array(data[k]) for k in data.columns if k.startswith("dingo:")}
    percentiles_is = {k[len("dingo-is:"):]: np.array(data[k]) for k in data.columns if k.startswith("dingo-is:")}

    make_pp_plot(np.stack(list(percentiles_is.values())).T, list(percentiles_is.keys()), os.path.join(outdir, "pp-is.pdf"))
    make_pp_plot(np.stack(list(percentiles.values())).T, list(percentiles.keys()), os.path.join(outdir, "pp.pdf"))

    fig, ax = plt.subplots(figsize=(10,10))
    # plt.hist(data['sample_efficiencies'] * 100, bins=100)
    plot_loghist(data['sample_efficiencies'] * 100, bins=100)
    plt.xlabel("Sample efficiency [%]")
    plt.ylabel("Count")
    plt.savefig(os.path.join(outdir, "sample_efficiencies.pdf"))
    print("Sample efficiency percentiles:")
    for p in [1, 5, 10, 20, 50]:
        print(f"{p:2.0f}: {np.percentile(data['sample_efficiencies'], p) * 100:5.1f}%")

    fig, ax = plt.subplots(figsize=(10,10))
    for k in data.columns:
        if k.startswith("chirp-mass-percentile-"):
            p = k.strip("chirp-mass-percentile-")
            plt.hist(data[k], bins=100, label=k)
    plt.xlabel("Delta chirp mass percentile")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(outdir, "chirp_mass_percentiles.pdf"))

