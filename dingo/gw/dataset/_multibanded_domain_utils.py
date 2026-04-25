"""Shared utilities for generating and evaluating MultibandedFrequencyDomain settings."""

from copy import deepcopy
from typing import Dict

import numpy as np

from dingo.gw.prior import build_prior_with_defaults


def build_extreme_prior(settings: dict):
    """Build a BBH prior with extreme parameter values to stress-test multibanding.

    Fixes ``chirp_mass`` to its minimum prior value and ``geocent_time`` to 0.12 s
    (the typical prior boundary plus the Earth-radius light-crossing time). All other
    parameters are sampled from the distributions specified in ``settings``.

    Parameters
    ----------
    settings : dict
        Dataset settings dict containing an ``'intrinsic_prior'`` key. Not modified.

    Returns
    -------
    BBHPriorDict
        Prior with extreme fixed values for ``chirp_mass`` and ``geocent_time``.
    """
    nominal_prior = build_prior_with_defaults(settings["intrinsic_prior"])
    extreme_settings = deepcopy(settings["intrinsic_prior"])
    extreme_settings["geocent_time"] = 0.12
    extreme_settings["chirp_mass"] = nominal_prior["chirp_mass"].minimum
    return build_prior_with_defaults(extreme_settings)


def print_mismatch_stats(mismatches: np.ndarray, num_samples: int) -> None:
    """Print a summary of mismatch statistics to stdout.

    Parameters
    ----------
    mismatches : np.ndarray
        1D array of mismatch values across all polarisations and samples.
    num_samples : int
        Number of waveform samples used, reported in the header line.
    """
    print("\nMismatches between UFD waveforms and MFD waveforms interpolated to UFD.")
    print(
        "This is a conservative estimate of the MFD performance when training networks."
    )
    print(f"num_samples = {num_samples}")
    print(f"  Mean mismatch = {np.mean(mismatches)}")
    print(f"  Standard deviation = {np.std(mismatches)}")
    print(f"  Max mismatch = {np.max(mismatches)}")
    print(f"  Median mismatch = {np.median(mismatches)}")
    print("  Percentiles:")
    print(f"    99    -> {np.percentile(mismatches, 99)}")
    print(f"    99.9  -> {np.percentile(mismatches, 99.9)}")
    print(f"    99.99 -> {np.percentile(mismatches, 99.99)}")
