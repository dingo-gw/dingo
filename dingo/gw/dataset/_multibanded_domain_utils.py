"""Shared utilities for generating and evaluating MultibandedFrequencyDomain settings."""

from copy import deepcopy
from typing import Dict

import numpy as np

from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import factor_fiducial_waveform


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
    # Pin the chirp mass to (essentially) its minimum -- the longest, hardest-to-decimate
    # signal. A bare scalar would become a bilby DeltaFunction, which breaks the
    # *constrained* sampling required by the mass_1/mass_2 Constraint priors (it raises
    # "non-broadcastable output operand with shape ()" in PriorDict.sample). A
    # negligibly narrow Uniform samples cleanly while keeping every draw at the minimum.
    mc_min = nominal_prior["chirp_mass"].minimum
    extreme_settings["chirp_mass"] = (
        f"bilby.core.prior.Uniform(minimum={mc_min}, maximum={mc_min * (1 + 1e-9)})"
    )
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


def heterodyne_polarizations(
    polarizations: Dict[str, np.ndarray],
    domain,
    parameters,
    settings: dict,
    chirp_mass_proxy_offset: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Heterodyne generated waveforms as the network input is, when the dataset
    settings request `phase_heterodyning` under `compression` (DINGO-BNS).

    A chirp-mass-conditioned network sees data heterodyned at the *proxy*, which
    differs from the true chirp mass by up to the width of the GNPE kernel; the
    residual oscillation, which sets the decimation, grows with that offset. The
    waveforms are therefore heterodyned at ``chirp_mass +- chirp_mass_proxy_offset``,
    the edges of the kernel, with the sign alternating by row: the offset term of the
    residual phase flips sign with the offset and adds to or cancels against the
    post-Newtonian remainder, so the two sides of the kernel are decimated
    differently and both must be represented. Without `phase_heterodyning` the
    waveforms are returned unchanged.

    Parameters
    ----------
    polarizations : Dict[str, np.ndarray]
        Waveforms on ``domain``, shape ``(num_samples, len(domain()))`` per key.
    domain
        Frequency domain of the waveforms.
    parameters : pd.DataFrame
        Waveform parameters, one row per sample (``chirp_mass``, and ``mass_ratio``
        for second-order heterodyning).
    settings : dict
        Dataset settings.
    chirp_mass_proxy_offset : float
        Magnitude of the offset of the heterodyne chirp mass from the true one, in
        solar masses. Default: 0.

    Returns
    -------
    Dict[str, np.ndarray]
        Heterodyned (or unchanged) waveforms.
    """
    heterodyning = settings.get("compression", {}).get("phase_heterodyning")
    if heterodyning is None:
        return polarizations
    sign = (-1.0) ** np.arange(len(parameters))
    chirp_mass = parameters["chirp_mass"].to_numpy() + sign * chirp_mass_proxy_offset
    mass_ratio = (
        parameters["mass_ratio"].to_numpy() if "mass_ratio" in parameters else None
    )
    return {
        k: factor_fiducial_waveform(v, domain, chirp_mass, mass_ratio, **heterodyning)
        for k, v in polarizations.items()
    }
