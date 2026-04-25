"""
Generate a MultibandedFrequencyDomain (MFD) settings file from a uniform frequency domain
(UFD) settings file by automatically tuning a decimation threshold to meet a target median
waveform mismatch.

The core idea is to use an extreme prior (minimum chirp mass, boundary geocent_time) to
stress-test the decimation, generate waveforms once, then binary-search over the whitened
waveform difference threshold until the desired mismatch level is reached.

CLI usage::

    python -m dingo.gw.dataset.generate_multibanded_domain \\
        --settings-file path/to/settings_wfd_ufd.yaml \\
        --num-samples 1000 \\
        --target-median-mismatch 0.001
"""

import argparse
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from bilby.gw.detector import PowerSpectralDensity
from scipy.interpolate import interp1d

from dingo.gw.dataset._multibanded_domain_utils import (build_extreme_prior,
                                                        print_mismatch_stats)
from dingo.gw.dataset.generate_dataset import \
    generate_parameters_and_polarizations
from dingo.gw.domains import (MultibandedFrequencyDomain,
                              UniformFrequencyDomain, build_domain)
from dingo.gw.gwutils import get_mismatch
from dingo.gw.waveform_generator import NewInterfaceWaveformGenerator


def floor_to_power_of_2(x: float) -> float:
    """Return the largest power of 2 that is <= x.

    Parameters
    ----------
    x : float
        Positive input value.

    Returns
    -------
    float
        Largest power of 2 not exceeding x.
    """
    return 2 ** (np.floor(np.log2(x)))


def compute_waveform_difference_per_decimation_factor(
    decimation_factors: np.ndarray,
    waveforms: np.ndarray,
    ufd: UniformFrequencyDomain,
    waveforms_2x: np.ndarray,
    difference_over_full_window: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute the 95th-percentile whitened waveform difference for each decimation factor.
    Only the real part of the waveform is considered.

    Two comparison modes are supported:

    - **Center comparison** (default, ``difference_over_full_window=False``): each
      decimated bin is compared to the waveform value at the center of the decimation
      window, read from the 2x-resolution reference array. This is less conservative
      but smoother.
    - **Full-window comparison** (``difference_over_full_window=True``): each decimated
      bin is compared to all original bins within the decimation window. This is the most
      conservative estimate.

    The resulting difference arrays are transformed into monotonically non-increasing
    sequences via a right-to-left cumulative maximum, reflecting the physical expectation
    that higher-frequency bins are easier to decimate accurately.

    Parameters
    ----------
    decimation_factors : np.ndarray
        Integer decimation factors to evaluate, e.g. ``2 ** np.arange(8)``.
    waveforms : np.ndarray
        Whitened real-part waveforms at base UFD resolution, shape
        ``(num_samples, num_bins)``.
    ufd : UniformFrequencyDomain
        Uniform frequency domain corresponding to ``waveforms``.
    waveforms_2x : np.ndarray
        Whitened real-part waveforms at twice the UFD resolution, used as the
        high-resolution reference in center-comparison mode.
    difference_over_full_window : bool
        If True, use full-window comparison. Default: False.

    Returns
    -------
    diffs : List[np.ndarray]
        One 1D array per decimation factor, containing the 95th-percentile
        whitened difference at each decimated frequency bin, monotonically
        non-increasing.
    freqs : List[np.ndarray]
        One 1D array per decimation factor, containing the corresponding
        reference frequencies for each entry in ``diffs``.
    """
    data = waveforms.real
    data_2x = waveforms_2x.real
    freq = ufd.sample_frequencies
    assert freq.shape[0] == data.shape[-1]

    freqs_per_decimation_factor: List[np.ndarray] = []
    diffs_per_decimation_factor: List[np.ndarray] = []

    for decimation_factor in decimation_factors:
        kernel = (
            torch.ones((1, 1, decimation_factor), dtype=torch.float64)
            / decimation_factor
        )

        def conv(
            d: np.ndarray, _k: torch.Tensor = kernel, _df: int = decimation_factor
        ) -> torch.Tensor:
            return torch.nn.functional.conv1d(
                torch.tensor(d, dtype=torch.float64)[:, None, :],
                _k,
                padding=0,
                stride=_df,
            ).squeeze()

        data_decimated = conv(data).numpy()

        if difference_over_full_window:
            f_ref: np.ndarray = freq
            data_ref: np.ndarray = data
            data_decimated = np.repeat(data_decimated, decimation_factor, axis=-1)
            n_pad = data_ref.shape[1] - data_decimated.shape[1]
            data_decimated = np.pad(
                data_decimated, ((0, 0), (0, n_pad)), "constant", constant_values=0
            )
        else:
            f_ref_tensor = conv(freq[None, :])
            inds_ref = (f_ref_tensor / (ufd.delta_f / 2)).type(torch.int32)
            data_ref = data_2x[:, inds_ref]
            f_ref = f_ref_tensor.numpy()

        diff = np.abs(data_ref - data_decimated)
        diff_perc = np.percentile(diff, 95, axis=0)
        diff_perc = np.maximum.accumulate(diff_perc[..., ::-1], axis=-1)[..., ::-1]
        diffs_per_decimation_factor.append(diff_perc)
        freqs_per_decimation_factor.append(f_ref)

    return diffs_per_decimation_factor, freqs_per_decimation_factor


def compute_max_decimation_factor(
    decimation_factors: np.ndarray,
    diffs_per_decimation_factor: List[np.ndarray],
    frequencies_per_decimation_factor: List[np.ndarray],
    ufd: UniformFrequencyDomain,
    threshold: float,
) -> np.ndarray:
    """Determine the maximum permitted decimation factor for each UFD frequency bin.

    Starting from a decimation factor of 1 for all bins, the function iterates over
    increasing decimation factors. For each, it finds the lowest frequency above which
    the 95th-percentile waveform difference falls below ``threshold``, then assigns that
    decimation factor to all bins above that frequency. If the transition frequency for a
    higher decimation factor is not strictly above the previous one, the search stops.

    Parameters
    ----------
    decimation_factors : np.ndarray
        Integer decimation factors in strictly increasing order.
    diffs_per_decimation_factor : List[np.ndarray]
        Per-decimation-factor whitened difference arrays, as returned by
        :func:`compute_waveform_difference_per_decimation_factor`.
    frequencies_per_decimation_factor : List[np.ndarray]
        Corresponding reference frequency arrays of the same structure.
    ufd : UniformFrequencyDomain
        Base uniform frequency domain.
    threshold : float
        Maximum permitted whitened waveform difference. Higher values allow more
        aggressive decimation.

    Returns
    -------
    max_dec_factor : np.ndarray
        Array of shape ``(len(ufd()),)`` with the maximum allowed decimation factor
        per UFD bin, monotonically non-decreasing.
    """
    max_dec_factor = np.ones(len(ufd()))
    f_max_threshold: List[float] = []

    for decimation_factor, f, diff in zip(
        decimation_factors,
        frequencies_per_decimation_factor,
        diffs_per_decimation_factor,
    ):
        f_max = f[np.argmax(diff < threshold)]
        if f_max_threshold:
            if f_max_threshold[-1] >= f_max:
                print(
                    f"Not possible to go to higher decimation factors than "
                    f"{max_dec_factor.max()} above {f_max_threshold[-1]}"
                )
                break
        f_max_threshold.append(float(f_max))
        max_dec_factor[int(f_max / ufd.delta_f) :] = decimation_factor

    return max_dec_factor


def get_band_nodes_for_adaptive_decimation(
    max_dec_factor_array: np.ndarray,
    max_dec_factor_global: int = np.inf,
) -> Tuple[int, List[int]]:
    """Convert a per-bin maximum decimation factor array into MFD band nodes.

    Iterates over the domain using the largest power-of-2 decimation factor permitted
    by ``max_dec_factor_array``, doubling the decimation factor each time the remaining
    bins allow it, until the entire domain is partitioned.

    Parameters
    ----------
    max_dec_factor_array : np.ndarray
        1D array of maximum allowed decimation factors per bin, monotonically
        non-decreasing.
    max_dec_factor_global : int
        Global upper bound on the decimation factor. Default: ``np.inf`` (no bound).

    Returns
    -------
    initial_downsampling : int
        Decimation factor of band 0.
    band_nodes : List[int]
        Bin-index boundaries of the bands. Band *j* spans
        ``[band_nodes[j], band_nodes[j+1])``. The first element is always 0.
    """
    if len(max_dec_factor_array.shape) != 1:
        raise ValueError("max_dec_factor_array needs to be 1D array.")
    if not (max_dec_factor_array[1:] >= max_dec_factor_array[:-1]).all():
        raise ValueError("max_dec_factor_array needs to increase monotonically.")

    max_dec_factor_array = np.clip(max_dec_factor_array, None, max_dec_factor_global)
    N = len(max_dec_factor_array)
    dec_factor = int(max(1.0, floor_to_power_of_2(float(max_dec_factor_array[0]))))
    band_nodes = [0]
    upper = dec_factor
    initial_downsampling = dec_factor

    while upper - 1 < N:
        if upper - 1 + dec_factor >= N:
            band_nodes.append(upper)
        elif dec_factor * 2 <= max_dec_factor_array[upper]:
            band_nodes.append(upper)
            dec_factor *= 2
        upper += dec_factor

    return initial_downsampling, band_nodes


def _get_asd_file(settings: dict) -> str:
    """Extract the ASD file path from dataset settings.

    Looks for the ASD under ``settings['compression']['whitening']``. Falls back to the
    standard aLIGO design-sensitivity ASD if the key is absent.

    Parameters
    ----------
    settings : dict
        Dataset settings dict, optionally containing a ``'compression'`` key with a
        ``'whitening'`` sub-key.

    Returns
    -------
    str
        Path or filename of the ASD file.
    """
    return settings.get("compression", {}).get(
        "whitening", "aLIGO_ZERO_DET_high_P_asd.txt"
    )


def _load_asd(
    asd_file: str,
    ufd_2x: UniformFrequencyDomain,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and interpolate an ASD file to both the UFD and 2x-resolution grids.

    The 1x ASD is obtained by taking every other sample of the 2x ASD, so only
    ``ufd_2x`` is needed. A spectral artefact (bump) in the frequency range 477–483 Hz
    is suppressed by replacing those values with the ASD value at 477 Hz, preventing it
    from artificially inflating the waveform difference in that band.

    Parameters
    ----------
    asd_file : str
        Path to the ASD file (bilby-compatible format).
    ufd_2x : UniformFrequencyDomain
        Domain at twice the resolution of the base UFD.

    Returns
    -------
    asd : np.ndarray
        ASD interpolated to ``ufd.sample_frequencies``.
    asd_2x : np.ndarray
        ASD interpolated to ``ufd_2x.sample_frequencies`` with bump removed.
    """
    psd = PowerSpectralDensity(asd_file=asd_file)
    asd_2x = np.interp(
        ufd_2x.sample_frequencies,
        psd.frequency_array,
        psd.asd_array,
        left=np.inf,
        right=np.inf,
    )
    bump_mask = np.where((ufd_2x() > 477) & (ufd_2x() < 483))[0]
    asd_2x[bump_mask] = asd_2x[np.argmin(np.abs(ufd_2x() - 477))]
    asd = asd_2x[::2]
    return asd, asd_2x


def _generate_whitened_waveforms(
    settings: dict,
    prior,
    asd_file: str,
    num_samples: int,
    num_processes: int = 1,
) -> Tuple[
    UniformFrequencyDomain,
    pd.DataFrame,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
]:
    """Generate waveforms at 2x resolution, whiten them, and return both 1x and 2x versions.

    Waveforms are generated at twice the base UFD resolution so that, when comparing a
    decimated waveform to its reference, the reference value at the center of each
    decimation window can be read from the high-resolution array (see
    :func:`compute_waveform_difference_per_decimation_factor`).

    Parameters
    ----------
    settings : dict
        Dataset settings containing ``'domain'`` and ``'waveform_generator'`` keys.
    prior
        Prior object used to draw waveform parameters.
    asd_file : str
        Path to the ASD file used for whitening.
    num_samples : int
        Number of waveforms to generate.
    num_processes : int
        Number of parallel processes for waveform generation. Default: 1.

    Returns
    -------
    ufd : UniformFrequencyDomain
        Base uniform frequency domain.
    parameters : pd.DataFrame
        Sampled waveform parameters.
    polarizations : Dict[str, np.ndarray]
        Raw (non-whitened) waveforms at 1x resolution, shape
        ``(num_samples, len(ufd()))``. Used for mismatch computation.
    data_whitened : Dict[str, np.ndarray]
        Whitened waveforms at 1x resolution. Used for band-node computation.
    data_whitened_2x : Dict[str, np.ndarray]
        Whitened waveforms at 2x resolution. Used as high-resolution reference.
    asd : np.ndarray
        ASD values at ``ufd.sample_frequencies``, used for pre-whitening in
        mismatch evaluation.
    """
    ufd = build_domain(settings["domain"])
    ufd_2x = build_domain({**ufd.domain_dict, **{"delta_f": ufd.delta_f / 2}})

    waveform_generator = NewInterfaceWaveformGenerator(
        domain=ufd_2x, **settings["waveform_generator"]
    )
    parameters, polarizations_2x = generate_parameters_and_polarizations(
        waveform_generator=waveform_generator,
        prior=prior,
        num_samples=num_samples,
        num_processes=num_processes,
    )

    polarizations = {k: v[..., ::2] for k, v in polarizations_2x.items()}
    assert polarizations["h_plus"][0].shape == ufd().shape

    asd, asd_2x = _load_asd(asd_file, ufd_2x)
    data_whitened = {k: v / asd / ufd.noise_std for k, v in polarizations.items()}
    data_whitened_2x = {
        k: v / asd_2x / ufd.noise_std for k, v in polarizations_2x.items()
    }

    return ufd, parameters, polarizations, data_whitened, data_whitened_2x, asd


def _build_mfd_for_threshold(
    diffs: List[np.ndarray],
    freqs: List[np.ndarray],
    decimation_factors: np.ndarray,
    ufd: UniformFrequencyDomain,
    threshold: float,
    delta_f_max_time_shift: float,
) -> MultibandedFrequencyDomain:
    """Construct a MultibandedFrequencyDomain from pre-computed differences and a threshold.

    Parameters
    ----------
    diffs : List[np.ndarray]
        Per-decimation-factor difference arrays from
        :func:`compute_waveform_difference_per_decimation_factor`.
    freqs : List[np.ndarray]
        Corresponding reference frequency arrays.
    decimation_factors : np.ndarray
        Decimation factors corresponding to entries in ``diffs`` and ``freqs``.
    ufd : UniformFrequencyDomain
        Base uniform frequency domain.
    threshold : float
        Whitened waveform difference threshold; higher values allow more aggressive
        decimation.
    delta_f_max_time_shift : float
        Maximum permitted frequency bin width (Hz) imposed by time-shift considerations.
        Sets ``max_dec_factor_global = delta_f_max_time_shift / ufd.delta_f``.

    Returns
    -------
    MultibandedFrequencyDomain
        Multibanded domain constructed for the given threshold.
    """
    max_dec_factor = compute_max_decimation_factor(
        decimation_factors, diffs, freqs, ufd, threshold
    )
    initial_downsampling, band_nodes_indices = get_band_nodes_for_adaptive_decimation(
        max_dec_factor[ufd.min_idx :],
        max_dec_factor_global=int(delta_f_max_time_shift / ufd.delta_f),
    )
    delta_f_initial = ufd.delta_f * initial_downsampling
    mfd_nodes = ufd()[ufd.min_idx :][np.array(band_nodes_indices)]
    return MultibandedFrequencyDomain(
        nodes=mfd_nodes, delta_f_initial=delta_f_initial, base_domain=ufd
    )


def _compute_mismatches(
    polarizations_ufd: Dict[str, np.ndarray],
    ufd: UniformFrequencyDomain,
    mfd: MultibandedFrequencyDomain,
    asd_ufd: np.ndarray,
) -> np.ndarray:
    """Compute mismatches between UFD waveforms and MFD-decimated versions interpolated back.

    For each waveform, the procedure is:

    1. Decimate the UFD waveform to the MFD via ``mfd.decimate``.
    2. Interpolate the MFD waveform back to UFD sample frequencies.
    3. Whiten both the original and interpolated waveforms with ``asd_ufd``.
    4. Compute the mismatch (1 - overlap) using inner products summed from ``ufd.min_idx``.

    This mirrors the evaluation in :func:`evaluate_multibanded_domain._evaluate_multibanding_main`
    while reusing pre-generated waveforms to avoid redundant waveform generation during
    threshold search.

    Parameters
    ----------
    polarizations_ufd : Dict[str, np.ndarray]
        Raw waveforms in the UFD, shape ``(num_samples, len(ufd()))``.
    ufd : UniformFrequencyDomain
        Base uniform frequency domain.
    mfd : MultibandedFrequencyDomain
        Candidate multibanded domain to evaluate.
    asd_ufd : np.ndarray
        ASD values at ``ufd.sample_frequencies``, shape ``(len(ufd()),)``, used
        for whitening prior to inner-product computation.

    Returns
    -------
    mismatches : np.ndarray
        1D array of mismatch values concatenated across all polarisations and samples.
    """
    ufd_freqs = ufd()
    mfd_freqs = mfd()

    all_mismatches: List[np.ndarray] = []
    for waveforms in polarizations_ufd.values():
        pol_mismatches = np.empty(len(waveforms))
        for i, wf_ufd in enumerate(waveforms):
            wf_mfd = mfd.decimate(wf_ufd)
            wf_mfd_interp = interp1d(mfd_freqs, wf_mfd, fill_value="extrapolate")(
                ufd_freqs
            )
            wf_ufd_white = wf_ufd / asd_ufd
            wf_mfd_interp_white = wf_mfd_interp / asd_ufd
            pol_mismatches[i] = get_mismatch(wf_ufd_white, wf_mfd_interp_white, ufd)
        all_mismatches.append(pol_mismatches)

    return np.concatenate(all_mismatches)


def _output_settings_path(settings_file: str) -> str:
    """Derive the output MFD settings file path from the input UFD settings path.

    Replaces the first occurrence of ``'_ufd'`` in the filename stem with ``'_mfd'``.
    If ``'_ufd'`` is not present, appends ``'_mfd'`` before the ``.yaml`` extension.
    The output is placed in the same directory as ``settings_file``.

    Parameters
    ----------
    settings_file : str
        Path to the input UFD settings YAML file.

    Returns
    -------
    str
        Absolute path for the output MFD settings YAML file.
    """
    directory = os.path.dirname(os.path.abspath(settings_file))
    basename = os.path.basename(settings_file)
    name, ext = os.path.splitext(basename)
    if "_ufd" in name:
        out_name = name.replace("_ufd", "_mfd", 1) + ext
    else:
        out_name = name + "_mfd" + ext
    return os.path.join(directory, out_name)


def generate_multibanded_domain_settings(
    settings_file: str,
    num_samples: int,
    target_median_mismatch: float,
    num_processes: int = 1,
    delta_f_max_time_shift: float = 2.0,
    decimation_factors: Optional[np.ndarray] = None,
    initial_threshold: float = 5e-3,
    max_iterations: int = 20,
) -> str:
    """Generate a MultibandedFrequencyDomain settings file targeting a given median mismatch.

    Loads a uniform frequency domain (UFD) settings file, generates waveforms once using
    an extreme prior (minimum chirp mass, boundary geocent time), then searches over the
    whitened waveform difference threshold until the MFD achieves the desired median
    mismatch. The resulting MFD settings are saved next to the input file, and mismatch
    statistics are printed to stdout.

    The search uses two phases:

    1. **Bracketing**: starting from ``initial_threshold``, walk geometrically outward
       (multiplying or dividing by a step factor that doubles each step) until the target
       mismatch is bracketed. This focuses evaluations near the likely solution rather than
       at extreme values that are unlikely to be close to the answer.
    2. **Bisection**: refine within the bracket until the MFD nodes converge (discrete
       structure) or the bracket width drops below 0.1%.

    Waveforms are generated only once and reused across all iterations, keeping the runtime
    cost proportional to a single waveform generation plus
    :math:`O(N_{\\text{iter}} \\cdot N_{\\text{samples}})` cheap operations.

    Parameters
    ----------
    settings_file : str
        Path to the UFD waveform dataset settings YAML. Must contain ``'domain'``,
        ``'waveform_generator'``, and ``'intrinsic_prior'`` keys.
    num_samples : int
        Number of waveforms used to determine the threshold and evaluate the mismatch.
    target_median_mismatch : float
        Desired upper bound on the median mismatch between UFD and MFD waveforms.
    num_processes : int
        Number of parallel processes for waveform generation. Default: 1.
    delta_f_max_time_shift : float
        Maximum permitted frequency bin width (Hz) set by time-shift requirements.
        Controls the global upper bound on the decimation factor. Default: 2.0.
    decimation_factors : np.ndarray, optional
        Decimation factors to evaluate. Default: ``2 ** np.arange(8)``.
    initial_threshold : float
        Starting point for the threshold search. Should be a reasonable central estimate;
        the search walks outward from here. Default: 5e-3.
    max_iterations : int
        Maximum number of iterations for each search phase. Default: 20.

    Returns
    -------
    output_path : str
        Path of the saved MFD settings YAML file.

    Raises
    ------
    RuntimeError
        If ``target_median_mismatch`` cannot be achieved even with the most conservative
        decimation reached during the downward walk.
    """
    if decimation_factors is None:
        decimation_factors = 2 ** np.arange(8)

    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    asd_file = _get_asd_file(settings)

    prior = build_extreme_prior(settings)
    print("Prior (extreme settings for stress-testing multibanding):")
    for k, v in prior.items():
        print(f"  {k}: {v}")

    print(f"\nGenerating {num_samples} waveforms at 2x resolution...")
    (
        ufd,
        _parameters,
        polarizations,
        data_whitened,
        data_whitened_2x,
        asd,
    ) = _generate_whitened_waveforms(
        settings, prior, asd_file, num_samples, num_processes
    )

    print("Computing waveform differences per decimation factor...")
    diffs, freqs = compute_waveform_difference_per_decimation_factor(
        decimation_factors=decimation_factors,
        waveforms=data_whitened["h_cross"],
        ufd=ufd,
        waveforms_2x=data_whitened_2x["h_cross"],
    )

    # --- Two-phase threshold search ---
    #
    # Phase 1 (bracketing): start at initial_threshold and walk geometrically outward,
    # doubling the step factor each iteration, until the target is bracketed. This avoids
    # evaluating at extreme values far from the solution.
    #
    # Phase 2 (bisection): refine within the bracket. Terminate early when both endpoints
    # map to identical MFD nodes (the threshold→nodes mapping is a step function).

    best_mfd: Optional[MultibandedFrequencyDomain] = None
    best_mismatches: Optional[np.ndarray] = None

    def _eval(t: float) -> Tuple[MultibandedFrequencyDomain, np.ndarray, float]:
        m = _build_mfd_for_threshold(
            diffs, freqs, decimation_factors, ufd, t, delta_f_max_time_shift
        )
        mis = _compute_mismatches(polarizations, ufd, m, asd)
        return m, mis, float(np.median(mis))

    def _log(label: str, t: float, med: float, m: MultibandedFrequencyDomain) -> None:
        comp = len(ufd()[ufd.frequency_mask]) / len(m())
        print(
            f"  {label:>12}  {t:>12.3e}  {med:>16.4e}  {m.num_bands:>6}  {comp:>11.1f}x"
        )

    print(
        f"\nSearching for threshold targeting median mismatch ≤ {target_median_mismatch:.2e}:"
    )
    print(
        f"  {'Step':>12}  {'Threshold':>12}  {'Median mismatch':>16}  "
        f"{'Bands':>6}  {'Compression':>12}"
    )

    # Evaluate starting point
    mfd_init, mis_init, med_init = _eval(initial_threshold)
    _log("init", initial_threshold, med_init, mfd_init)

    # Phase 1: bracketing walk
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None

    if med_init <= target_median_mismatch:
        # Initial threshold is feasible; walk upward to find where target is exceeded.
        best_mfd, best_mismatches = mfd_init, mis_init
        threshold_low = initial_threshold
        factor = 2.0
        for i in range(max_iterations):
            t = threshold_low * factor
            mfd_t, mis_t, med_t = _eval(t)
            _log(f"up {i + 1}", t, med_t, mfd_t)
            if med_t > target_median_mismatch:
                threshold_high = t
                break
            best_mfd, best_mismatches = mfd_t, mis_t
            threshold_low = t
            factor *= 2.0
        if threshold_high is None:
            print(
                "  Target achievable at all tested thresholds; using most aggressive."
            )
    else:
        # Initial threshold is too aggressive; walk downward to find where target is met.
        threshold_high = initial_threshold
        factor = 2.0
        for i in range(max_iterations):
            t = threshold_high / factor
            mfd_t, mis_t, med_t = _eval(t)
            _log(f"down {i + 1}", t, med_t, mfd_t)
            if med_t <= target_median_mismatch:
                threshold_low = t
                best_mfd, best_mismatches = mfd_t, mis_t
                break
            threshold_high = t
            factor *= 2.0
        if threshold_low is None:
            raise RuntimeError(
                f"Target median mismatch {target_median_mismatch:.2e} is too stringent: "
                f"even the smallest tested threshold yields median mismatch > target. "
                "Consider increasing the target or using more waveform samples."
            )

    # Phase 2: bisection within [threshold_low, threshold_high]
    if threshold_low is not None and threshold_high is not None:
        lo_nodes = _build_mfd_for_threshold(
            diffs, freqs, decimation_factors, ufd, threshold_low, delta_f_max_time_shift
        ).nodes
        hi_nodes = _build_mfd_for_threshold(
            diffs,
            freqs,
            decimation_factors,
            ufd,
            threshold_high,
            delta_f_max_time_shift,
        ).nodes
        if np.allclose(lo_nodes, hi_nodes):
            print("\n  Bracket maps to identical MFD nodes; no bisection needed.")
        else:
            print(f"\n  Refining bracket [{threshold_low:.3e}, {threshold_high:.3e}]:")
            previous_nodes: Optional[np.ndarray] = None
            for iteration in range(max_iterations):
                threshold = np.sqrt(threshold_low * threshold_high)
                mfd, mismatches, median_mismatch = _eval(threshold)
                _log(f"bisect {iteration + 1}", threshold, median_mismatch, mfd)

                if median_mismatch <= target_median_mismatch:
                    best_mfd, best_mismatches = mfd, mismatches
                    threshold_low = threshold
                else:
                    threshold_high = threshold

                if previous_nodes is not None and np.allclose(
                    mfd.nodes, previous_nodes
                ):
                    print("  Nodes unchanged; bisection converged.")
                    break
                previous_nodes = mfd.nodes.copy()

                if threshold_high / threshold_low < 1.001:
                    print("  Bracket width < 0.1%; bisection converged.")
                    break

    # --- Report and save ---
    mfd_final = best_mfd
    mismatches_final = best_mismatches

    print(f"\nFinal MFD:")
    print(f"  Nodes          : {mfd_final.nodes.tolist()}")
    print(f"  delta_f_initial: {mfd_final._delta_f_bands[0].item()}")
    print(f"  Num bands      : {mfd_final.num_bands}")
    print(
        f"  Compression    : {len(ufd()[ufd.frequency_mask]) / len(mfd_final()):.2f}x"
    )
    print_mismatch_stats(mismatches_final, num_samples)

    output_settings = deepcopy(settings)
    output_settings["domain"] = {
        "type": "MultibandedFrequencyDomain",
        "nodes": [float(f) for f in mfd_final.nodes],
        "delta_f_initial": float(mfd_final._delta_f_bands[0]),
        "base_domain": settings["domain"],
    }

    output_path = _output_settings_path(settings_file)
    with open(output_path, "w") as f:
        yaml.dump(output_settings, f)
    print(f"\nSaved MFD settings to: {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate a waveform dataset settings file with a MultibandedFrequencyDomain "
        "from a waveform dataset settings file with a UniformFrequencyDomain.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Path to the UFD waveform dataset settings YAML file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of waveforms used for threshold tuning and mismatch evaluation.",
    )
    parser.add_argument(
        "--target_median_mismatch",
        type=float,
        required=True,
        help="Desired upper bound on the median mismatch between UFD and MFD waveforms.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes for waveform generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the generate_multibanded_domain CLI."""
    args = parse_args()
    generate_multibanded_domain_settings(
        settings_file=args.settings_file,
        num_samples=args.num_samples,
        target_median_mismatch=args.target_median_mismatch,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
