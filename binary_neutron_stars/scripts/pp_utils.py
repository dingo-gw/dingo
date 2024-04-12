import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import argparse
import textwrap


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


def make_pp_plot(percentiles, parameter_labels, filename, ks=True,
            confidence_interval=[0.68, 0.95, 0.997], confidence_interval_alpha=0.1):
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
        
    fig, ax = plt.subplots(figsize=(10,10))
    
    for n in range(nparams):
        if ks:
            pvalue = stats.kstest(percentiles[:, n], 'uniform')[1]
            plt.step(ordered[:, n], y, where='post', label=parameter_labels[n] + r' ({:.3g})'.format(pvalue))
        else:
            plt.step(ordered[:, n], y, where='post', label=parameter_labels[n])

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
            ax.fill_between(y, lower, upper, alpha=alpha, color='k')
    
    plt.plot(y, y, 'k--')
    plt.legend()
    plt.xlabel(r'$p$')
    plt.ylabel(r'$CDF(p)$')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    ax.set_aspect('equal', anchor='SW')
    plt.savefig(filename)
    plt.show()