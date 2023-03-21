import numpy as np
import torch
import lal

from dingo.gw.domains import build_domain, FrequencyDomain


def get_decimation_bands_adaptive(
    original_domain: FrequencyDomain,
    waveforms: np.ndarray,
    min_num_bins_per_period: int = 8,
    delta_f_max: float = np.inf,
):
    """
    Get frequency bands for decimation, which can be used to initialize a
    MultibandedFrequencyDomain object. This is based on the waveforms array. First,
    the oscillation periods are extracted from the waveforms. Next, frequency bands are
    set up such that each oscillation is captured by at least min_num_bins_per_period
    bins. The decimation factor increases by a factor of 2 between consecutive bands.

    Parameters
    ----------
    original_domain: FrequencyDomain
        Original uniform frequency domain of the data to be decimated.
    waveforms: np.ndarray
        2D array with complex waveforms in the original uniform frequency domain. Used
        to determine the required resolution, and thereby the boundaries of the bands.
    min_num_bins_per_period: int = 8
        Minimum number of bins per oscillation period.
        Note: a period here describes the interval between two consecutive zero
        crossings, so it differs by a factor of 2 from the usual convention.
    delta_f_max: float = np.inf
        Maximum delta_f of the bands.

    Returns
    -------
    bands: list
        List of frequency bands. Can be used to initialize MultibandedFrequencyDomain.

    """
    if len(waveforms.shape) != 2 or waveforms.shape[-1] != len(original_domain):
        raise ValueError(
            f"Waveform array has shape {waveforms.shape}, expected, "
            f"(N, {len(original_domain)}): "
            f"N waveforms, each of the same length {len(original_domain)} as domain."
        )

    x = waveforms[:, original_domain.min_idx :].real
    max_dec_factor_global = delta_f_max / original_domain.delta_f
    # Ideally, we would just call
    #   periods = np.min(get_periods(x, upper_bound_for_monotonicity=True), axis=0)
    # here. However, get_periods does not work perfectly on phase-heterodyned BNS
    # waveforms. The reason is that get_periods assumes waveforms that oscillate
    # symmetrically around the origin. However, in practice there are some waveforms
    # for which the oscillation of the real part is not symmetric in some segment of the
    # frequency axis. Instead of an oscillation in the range (-a, a) [with a being the
    # local amplitude], we sometimes encounter oscillations in range (-eps, a) with
    # a >> eps > 0. I suspect that this behaviour is an artefact of the phase
    # heterodyning. Since get_periods infers the periods based on the zero crossings,
    # it will infer a very small period for that frequency segment. In these rare cases,
    # the inferred period is not a good approximation of the number of bins required to
    # capture the oscillation, as the rate of change of the signal is much smaller than
    # what the period suggests. So below, we remove these cases by using
    # np.percentile(_, 1) instead of np.min(_).
    periods = np.percentile(
        get_periods(x, upper_bound_for_monotonicity=False), 1, axis=0
    )
    periods = np.minimum.accumulate(periods[::-1])[::-1]
    max_dec_factor_array = periods / min_num_bins_per_period
    bands_inds = get_inds_for_adaptive_decimation(
        max_dec_factor_array, max_dec_factor_global
    )

    # transform the indices and decimation factors into the frequency bounds and
    # delta_f for the bands.
    f = original_domain()[original_domain.min_idx :]
    bands = []
    for lower, upper, dec_factor in bands_inds:
        delta_f_band = dec_factor * original_domain.delta_f
        f_min_band = f[lower] + (dec_factor - 1) / 2 * original_domain.delta_f
        f_max_band = f[upper] - (dec_factor - 1) / 2 * original_domain.delta_f
        bands.append([f_min_band, f_max_band, delta_f_band])

    # test consistency
    # mfd = MultibandedFrequencyDomain(bands, original_domain)
    # bands_inds_old = np.array(bands_inds)
    # bands_inds_old[..., :2] += original_domain.min_idx
    # assert (bands_inds_old == np.array(mfd.decimation_inds_bands)).all()

    return bands


def get_decimation_bands_from_chirp_mass(
    original_domain: FrequencyDomain,
    chirp_mass_min: float,
    alpha: int = 1,
    delta_f_max: float = np.inf,
):
    """
    Get frequency bands for decimation, which can be used to initialize a
    MultibandedFrequencyDomain object. This is based on the minimal chirp mass,
    which to leading order determines the required frequency resolution in each
    frequency band.

    Parameters
    ----------
    original_domain: FrequencyDomain
        Original uniform frequency domain of the data to be decimated.
    chirp_mass_min: float
        Minimum chirp mass. Smaller chirp masses require larger resolution.
    alpha: int
        Factor by which to decrease the resolution. Needs to be a power of 2.
        The resolution can for instance be decreased when using heterodyning.
    delta_f_max: float = np.inf
        Maximum delta_f of the bands.

    Returns
    -------
    bands: list
        List of frequency bands. Can be used to initialize MultibandedFrequencyDomain.
    """
    if not is_power_of_2(1 / original_domain.delta_f):
        raise NotImplementedError(
            f"Decimation only implemented for domains with delta_f = 1 / k**2, "
            f"got {original_domain.delta_f}."
        )
    if not is_power_of_2(alpha):
        raise NotImplementedError(f"Alpha needs to be a power of 2, got {alpha}.")

    # delta_f and f_min for first band, derived from original_domain and chirp_mass_min
    delta_f_band = alpha / ceil_to_power_of_2(
        duration_LO(chirp_mass_min, original_domain.f_min)
    )
    # delta_f can't be smaller than original_domain.delta_f
    delta_f_band = max(delta_f_band, original_domain.delta_f)
    f = original_domain.f_min - original_domain.delta_f / 2.0 + delta_f_band / 2.0
    bands = []

    while f + delta_f_band / 2 < original_domain.f_max:

        f_min_band = f
        while is_within_band(
            f + delta_f_band,
            chirp_mass_min,
            delta_f_band,
            original_domain.f_max,
            alpha,
            delta_f_max,
        ):
            f += delta_f_band
        f_max_band = f
        bands.append([f_min_band, f_max_band, delta_f_band])

        delta_f_band *= 2
        f += (delta_f_band / 2 + delta_f_band) / 2

    return bands


def decimate_uniform(data, decimation_factor: int):
    """
    Reduce dimension of data by decimation_factor along last axis, by uniformly
    averaging sets of decimation_factor neighbouring bins.

    Parameters
    ----------
    data
        Array or tensor to be decimated.
    decimation_factor
        Factor by how much to compress. Needs to divide data.shape[-1].
    Returns
    -------
    data_decimated
        Uniformly decimated data, as array or tensor.
        Shape (*data.shape[:-1], data.shape[-1]/decimation_factor).
    """
    if data.shape[-1] % decimation_factor != 0:
        raise ValueError(
            f"data.shape[-1] ({data.shape[-1]} is not a multiple of decimation_factor "
            f"({decimation_factor})."
        )
    if isinstance(data, np.ndarray):
        return (
            np.sum(np.reshape(data, (*data.shape[:-1], -1, decimation_factor)), axis=-1)
            / decimation_factor
        )
    elif isinstance(data, torch.Tensor):
        return (
            torch.sum(
                torch.reshape(data, (*data.shape[:-1], -1, decimation_factor)), dim=-1
            )
            / decimation_factor
        )
    else:
        raise NotImplementedError(
            f"Decimation not implemented for data of type {data}."
        )


def ceil_to_power_of_2(x):
    return 2 ** (np.ceil(np.log2(x)))


def floor_to_power_of_2(x):
    return 2 ** (np.floor(np.log2(x)))


def is_power_of_2(x):
    return 2 ** int(np.log2(x)) == x


def duration_LO(chirp_mass, frequency):
    # Eq. (3) in https://arxiv.org/abs/1703.02062
    # in geometric units:
    f = frequency / lal.C_SI
    M = chirp_mass * lal.GMSUN_SI / lal.C_SI ** 2
    t = 5 * (8 * np.pi * f) ** (-8 / 3) * M ** (-5 / 3)
    return t / lal.C_SI


def is_within_band(f, chirp_mass_min, delta_f_band, f_max, alpha=1, delta_f_max=np.inf):
    # check if next frequency value would be larger than global f_max
    if f + delta_f_band / 2 > f_max:
        return False
    # check whether delta_f can be increased (if yes, return False)
    elif (
        duration_LO(chirp_mass_min, f) < 1 / (2 * delta_f_band * alpha)
        and 2 * delta_f_band <= delta_f_max
    ):
        return False
    else:
        return True


def number_of_zero_crossings(x):
    if np.iscomplex(x).any():
        raise ValueError("Only works for real arrays.")
    return np.sum((x[..., :-1] * x[..., 1:]) < 0, axis=-1) + np.sum(x == 0, axis=-1)


def get_periods(x: np.ndarray, upper_bound_for_monotonicity: bool = False):
    """
    Takes 1D or 2D array x as input. Returns array of the same shape, specifying the
    cycle length for each bin (axis=-1). This is done by checking for zero-crossings
    in x. The lower/upper boundaries are filled with the periods from the neighboring
    intervals.

    Note: This assumes an oscillatory behavior of x about 0.
    Note: A period here describes the interval between two consecutive zero crossings,
    so it differs by a factor of 2 from the usual convention.

    Parameters
    ----------
    x: np.ndarray
        Array with oscillation signal. 1D or 2D, oscillation pattern on axis -1.
    upper_bound_for_monotonicity: bool = False
        If set, then the periods returned increase monotonically.

    Returns
    -------
    periods_expanded: np.ndarray
        Array with same shape as x, containing the period (as int) for each bin.

    Examples
    --------
    >>> x = np.array([-1, 0, 1, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1])
    >>> get_periods(x)
    array([4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6])
    >>> get_periods(x, upper_bound_for_monotonicity=True)
    array([4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6])

    >>> x = np.array([-1, 0, 1, 2, 1, 0, -1, 0, 1])
    >>> get_periods(x)
    array([4, 4, 4, 4, 4, 2, 2, 2, 2])
    >>> get_periods(x, upper_bound_for_monotonicity=True)
    array([2, 2, 2, 2, 2, 2, 2, 2, 2])
    """
    if np.iscomplex(x).any():
        raise ValueError("Only works for real arrays.")

    # implementation for single arrays
    if len(x.shape) == 1:
        zero_crossings = np.where(
            (x[:-1] >= 0) & (x[1:] < 0) | (x[:-1] <= 0) & (x[1:] > 0)
        )[0]
        periods_expanded = np.zeros(len(x), dtype=int)
        for lower, upper in zip(zero_crossings[:-1], zero_crossings[1:]):
            periods_expanded[lower:upper] = upper - lower
        # fill in boundaries
        periods_expanded[: zero_crossings[0]] = periods_expanded[zero_crossings[0]]
        periods_expanded[zero_crossings[-1] :] = periods_expanded[
            zero_crossings[-1] - 1
        ]
        # if monotonically increasing periods are requested, upper bound the periods
        # with periods_expanded[i] = min(periods_expanded[i:]).
        if upper_bound_for_monotonicity:
            periods_expanded = np.minimum.accumulate(periods_expanded[::-1])[::-1]
        return periods_expanded

    # batched arrays: recurse with single arrays (there might be a faster way to do this)
    elif len(x.shape) == 2:
        periods_expanded = np.empty(x.shape, dtype=int)
        for idx, x_single in enumerate(x):
            periods_expanded[idx, :] = get_periods(
                x_single, upper_bound_for_monotonicity
            )
        return periods_expanded

    else:
        raise ValueError(
            f"Only implemented for single or batched arrays, got {len(x.shape)} axes."
        )


def get_inds_for_adaptive_decimation(
    max_dec_factor_array: np.ndarray, max_dec_factor_global: int = np.inf
):
    """
    Sets up adaptive multibanding for decimation. The 1D array max_dec_factor_array has
    the same length as the original, and contains the maximal acceptable decimation
    factors for each bin. max_dec_factor_global further specifies the maximum
    decimation factor.

    Parameters
    ----------
    max_dec_factor_array: np.ndarray
        Array with maximal decimation factor for each bin. Monotonically increasing.
    max_dec_factor_global: int = np.inf
        Global maximum for decimation factor.

    Returns
    -------
    bands: list
        List with decimation indices, in format [lower, upper, dec_factor] for each band.
        (lower, upper) are the *inclusive* boundaries for the band.
    """
    if len(max_dec_factor_array.shape) != 1:
        raise ValueError("max_dec_factor_array needs to be 1D array.")
    if not (max_dec_factor_array[1:] >= max_dec_factor_array[:-1]).all():
        raise ValueError("max_dec_factor_array needs to increase monotonically.")

    bands = []
    max_dec_factor_array = np.clip(max_dec_factor_array, None, max_dec_factor_global)
    N = len(max_dec_factor_array)
    dec_factor = int(max(1, floor_to_power_of_2(max_dec_factor_array[0])))
    lower_band = 0
    upper = lower_band + dec_factor - 1
    while upper < N:
        if upper + dec_factor >= N:
            bands.append([lower_band, upper, dec_factor])
        elif dec_factor * 2 <= max_dec_factor_array[upper + 1]:
            # conclude previous band
            bands.append([lower_band, upper, dec_factor])
            # enter new band
            dec_factor *= 2
            lower_band = upper + 1
        upper += dec_factor

    return bands
