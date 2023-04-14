from typing import Iterable, Union
import numpy as np
import torch
import lal

from .base import Domain
from .frequency_domain import FrequencyDomain


class MultibandedFrequencyDomain(Domain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max]
    with spacing delta_f.
    Given a finite length of time domain data, the Fourier domain data
    starts at a frequency f_min and is zero below this frequency.
    window_kwargs specify windowing used for FFT to obtain FD data from TD
    data in practice.
    """

    def __init__(
        self,
        nodes: Iterable[float],
        delta_f_initial: float,
        base_domain: Union[FrequencyDomain, dict],
        window_factor: float = None,
    ):
        """
        Parameters
        ----------
        nodes: Iterable[float]
            Frequency nodes for bands.
            In total, there are len(nodes) - 1 frequency bands.
            Band j consists of decimated data from the base domain in the range
            [nodes[j]:nodes[j+1].
        delta_f_initial: float
            delta_f of band 0. The decimation factor doubles between adjacent bands,
            so delta_f is halved.
        base_domain: Union[FrequencyDomain, dict]
            Original (uniform frequency) domain of data, which is the starting point
            for the decimation. This determines the decimation details and the noise_std.
            Either provided as dict for build_domain, or as domain_object.
        window_factor: float = None
            Window factor for this domain. Required when using self.noise_std.
        """
        if type(base_domain) == dict:
            from dingo.gw.domains import build_domain

            base_domain = build_domain(base_domain)

        self._window_factor = window_factor
        self.nodes = np.array(nodes)
        self.base_domain = base_domain
        self.initialize_bands(delta_f_initial)
        if not isinstance(self.base_domain, FrequencyDomain):
            raise ValueError(
                f"Expected domain type FrequencyDomain, got {type(base_domain)}."
            )

    def initialize_bands(self, delta_f_initial):
        if len(self.nodes.shape) != 1:
            raise ValueError(
                f"Expected format [num_bands + 1] for nodes, "
                f"got {self.nodes.shape}."
            )
        self.num_bands = len(self.nodes) - 1
        self.nodes_indices = (self.nodes / self.base_domain.delta_f).astype(int)
        self._delta_f_bands = delta_f_initial * (2 ** np.arange(self.num_bands))

        self._decimation_factors_bands = (
            self._delta_f_bands / self.base_domain.delta_f
        ).astype(int)
        self._num_bins_bands = (
            (self.nodes_indices[1:] - self.nodes_indices[:-1])
            / self._decimation_factors_bands
        ).astype(int)

        self._sample_frequencies = np.zeros(np.sum(self._num_bins_bands))
        self._sample_frequencies = self.decimate(self.base_domain())
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None
        # array with delta_f for each bin
        self._delta_f = np.concatenate(
            [
                np.ones(n) * delta_f
                for n, delta_f in zip(self._num_bins_bands, self._delta_f_bands)
            ]
        )

    def decimate(self, data):
        if data.shape[-1] == len(self.base_domain):
            offset_idx = 0
        elif data.shape[-1] == len(self.base_domain) - self.base_domain.min_idx:
            offset_idx = -self.base_domain.min_idx
        else:
            raise ValueError(
                f"Provided data has {data.shape[-1]} bins, which is incompatible with "
                f"the expected domain of length {len(self.base_domain)}"
            )
        if isinstance(data, np.ndarray):
            data_decimated = np.empty((*data.shape[:-1], len(self)), dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            data_decimated = torch.empty(
                (*data.shape[:-1], len(self)), dtype=data.dtype
            )
        else:
            raise NotImplementedError(
                f"Decimation not implemented for data of type {data}."
            )

        lower_out = 0  # running index for decimated band data
        for idx_band in range(self.num_bands):
            lower_in = self.nodes_indices[idx_band] + offset_idx
            upper_in = self.nodes_indices[idx_band + 1] + offset_idx
            decimation_factor = self._decimation_factors_bands[idx_band]
            num_bins = self._num_bins_bands[idx_band]

            data_decimated[..., lower_out : lower_out + num_bins] = decimate_uniform(
                data[..., lower_in:upper_in], decimation_factor
            )
            lower_out = lower_out + num_bins

        assert lower_out == len(self)

        return data_decimated

    def update(self, new_settings: dict):
        if not new_settings == self.domain_dict:
            raise NotImplementedError()

    def set_new_range(self, f_min: float = None, f_max: float = None):
        raise NotImplementedError()

    def update_data(self, data: np.ndarray, axis: int = -1, low_value: float = 0.0):
        """
        Adjusts the data to be compatible with the domain. Updating
        MultibandedFrequencyDomains is not implemented, so at present this method
        simply checks that the data is compatible with the domain.
        """
        if data.shape[axis] != len(self):
            raise ValueError(
                f"Data (shape {data.shape}) incompatible with the domain (length "
                f"{len(self)}."
            )
        return data

    def time_translate_data(self, data, dt):
        """
        TODO: like self.add_phase, this is just copied from FrequencyDomain and
        TODO: could be inherited instead.
        Time translate frequency-domain data by dt. Time translation corresponds (in
        frequency domain) to multiplication by

        .. math::
            \exp(-2 \pi i \, f \, dt).

        This method allows for multiple batch dimensions. For torch.Tensor data,
        allow for either a complex or a (real, imag) representation.

        Parameters
        ----------
        data : array-like (numpy, torch)
            Shape (B, C, N), where

                - B corresponds to any dimension >= 0,
                - C is either absent (for complex data) or has dimension >= 2 (for data
                  represented as real and imaginary parts), and
                - N is either len(self) or len(self)-self.min_idx (for truncated data).

        dt : torch tensor, or scalar (if data is numpy)
            Shape (B)

        Returns
        -------
        Array-like of the same form as data.
        """
        f = self.get_sample_frequencies_astype(data)
        if isinstance(data, np.ndarray):
            # Assume numpy arrays un-batched, since they are only used at train time.
            phase_shift = 2 * np.pi * dt * f
        elif isinstance(data, torch.Tensor):
            # Allow for possible multiple "batch" dimensions (e.g., batch + detector,
            # which might have independent time shifts).
            phase_shift = 2 * np.pi * torch.einsum("...,i", dt, f)
        else:
            raise NotImplementedError(
                f"Time translation not implemented for data of " "type {data}."
            )
        return self.add_phase(data, phase_shift)

    def get_sample_frequencies_astype(self, data):
        """
        Returns a 1D frequency array compatible with the last index of data array.

        Decides whether array is numpy or torch tensor (and cuda vs cpu).

        Parameters
        ----------
        data : Union[np.array, torch.Tensor]
            Sample data

        Returns
        -------
        frequency array compatible with last index
        """
        # Type
        if isinstance(data, np.ndarray):
            f = self.sample_frequencies
        elif isinstance(data, torch.Tensor):
            if data.is_cuda:
                f = self.sample_frequencies_torch_cuda
            else:
                f = self.sample_frequencies_torch
        else:
            raise TypeError("Invalid data type. Should be np.array or torch.Tensor.")

        return f

    @staticmethod
    def add_phase(data, phase):
        """
        TODO: Copied from FrequencyDomain. Should this be inherited instead?
        TODO: Maybe there should be a shared parent class FrequencyDomain, that
        TODO: UniformFrequencyDomain and MultibandedFrequencyDomain inherit from.

        Add a (frequency-dependent) phase to a frequency series. Allows for batching,
        as well as additional channels (such as detectors). Accounts for the fact that
        the data could be a complex frequency series or real and imaginary parts.

        Convention: the phase phi(f) is defined via exp(- 1j * phi(f)).

        Parameters
        ----------
        data : Union[np.array, torch.Tensor]
        phase : Union[np.array, torch.Tensor]

        Returns
        -------
        New array or tensor of the same shape as data.
        """
        if isinstance(data, np.ndarray) and np.iscomplexobj(data):
            # This case is assumed to only occur during inference, with un-batched data.
            return data * np.exp(-1j * phase)

        elif isinstance(data, torch.Tensor):
            if torch.is_complex(data):
                # Expand the trailing batch dimensions to allow for broadcasting.
                while phase.dim() < data.dim():
                    phase = phase[..., None, :]
                return data * torch.exp(-1j * phase)
            else:
                # The first two components of the second last index should be the real
                # and imaginary parts of the data. Remaining components correspond to,
                # e.g., the ASD. The "-1" below accounts for this extra dimension when
                # broadcasting.
                while phase.dim() < data.dim() - 1:
                    phase = phase[..., None, :]

                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                result = torch.empty_like(data)
                result[..., 0, :] = (
                    data[..., 0, :] * cos_phase + data[..., 1, :] * sin_phase
                )
                result[..., 1, :] = (
                    data[..., 1, :] * cos_phase - data[..., 0, :] * sin_phase
                )
                if data.shape[-2] > 2:
                    result[..., 2:, :] = data[..., 2:, :]
                return result

        else:
            raise TypeError(f"Invalid data type {type(data)}.")

    def __len__(self):
        """Number of frequency bins in the domain"""
        return len(self._sample_frequencies)

    def __call__(self) -> np.ndarray:
        """Array of multibanded frequency bins in the domain [f_min, f_max]"""
        return self.sample_frequencies

    def __getitem__(self, idx):
        """Slice of joint frequency grid."""
        raise NotImplementedError()

    @property
    def sample_frequencies(self):
        return self._sample_frequencies

    @property
    def sample_frequencies_torch(self):
        if self._sample_frequencies_torch is None:
            num_bins = len(self)
            self._sample_frequencies_torch = torch.linspace(
                0.0, self.f_max, steps=num_bins, dtype=torch.float32
            )
        return self._sample_frequencies_torch

    @property
    def sample_frequencies_torch_cuda(self):
        if self._sample_frequencies_torch_cuda is None:
            self._sample_frequencies_torch_cuda = self.sample_frequencies_torch.to(
                "cuda"
            )
        return self._sample_frequencies_torch_cuda

    @property
    def frequency_mask(self) -> np.ndarray:
        raise NotImplementedError()

    def _reset_caches(self):
        raise NotImplementedError()

    @property
    def frequency_mask_length(self) -> int:
        raise NotImplementedError()

    @property
    def min_idx(self):
        return 0

    @property
    def max_idx(self):
        raise NotImplementedError()

    @property
    def window_factor(self):
        return self._window_factor

    @window_factor.setter
    def window_factor(self, value):
        """Set self._window_factor."""
        self._window_factor = float(value)

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this array. In practice, this means
        dividing the whitened waveforms by this.

        In contrast to the uniform FrequencyDomain, this is an array and not a number,
        as self._delta_f is not constant.

        TODO: This description makes some assumptions that need to be clarified.
        Windowing of TD data; tapering window has a slope -> reduces power only for noise,
        but not for the signal which is in the main part unaffected by the taper
        """
        if self._window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std.")
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    def f_max(self) -> float:
        raise NotImplementedError()

    @f_max.setter
    def f_max(self, value):
        raise NotImplementedError()

    @property
    def f_min(self) -> float:
        raise NotImplementedError()

    @f_min.setter
    def f_min(self, value):
        raise NotImplementedError()

    @property
    def delta_f(self) -> float:
        raise NotImplementedError()

    @delta_f.setter
    def delta_f(self, value):
        raise NotImplementedError()

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError()

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        # Call tolist() on self.bands, such that it can be saved as str for metadata.
        return {
            "type": "MultibandedFrequencyDomain",
            "nodes": self.nodes.tolist(),
            "delta_f_initial": self._delta_f_bands[0].item(),
            "base_domain": self.base_domain.domain_dict,
            "window_factor": self.window_factor,
        }


######################
### util functions ###
######################


def get_decimation_bands_adaptive(
    base_domain: FrequencyDomain,
    waveforms: np.ndarray,
    min_num_bins_per_period: int = 16,
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
    base_domain: FrequencyDomain
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
    if len(waveforms.shape) != 2 or waveforms.shape[-1] != len(base_domain):
        raise ValueError(
            f"Waveform array has shape {waveforms.shape}, expected, "
            f"(N, {len(base_domain)}): "
            f"N waveforms, each of the same length {len(base_domain)} as domain."
        )

    # For some reason, the last bin of a waveform is always zero, so we need to get rid
    # of that for the step below.
    x = waveforms[:, base_domain.min_idx : -1]

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
    periods = get_periods(x.real, upper_bound_for_monotonicity=False)
    # periods = get_period_for_complex_oscillation(x, upper_bound_for_monotonicity=False)
    periods = np.percentile(periods, 1, axis=0)
    periods = np.minimum.accumulate(periods[::-1])[::-1]

    max_dec_factor_array = periods / min_num_bins_per_period
    initial_downsampling, band_nodes_indices = get_band_nodes_for_adaptive_decimation(
        max_dec_factor_array,
        max_dec_factor_global=int(delta_f_max / base_domain.delta_f),
    )

    # transform downsampling factor and band nodes from indices to frequencies
    delta_f_initial = base_domain.delta_f * initial_downsampling
    nodes = base_domain()[np.array(band_nodes_indices) + base_domain.min_idx]

    return nodes, delta_f_initial


def get_decimation_bands_from_chirp_mass(
    base_domain: FrequencyDomain,
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
    base_domain: FrequencyDomain
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
    if not is_power_of_2(1 / base_domain.delta_f):
        raise NotImplementedError(
            f"Decimation only implemented for domains with delta_f = 1 / k**2, "
            f"got {base_domain.delta_f}."
        )
    if not is_power_of_2(alpha):
        raise NotImplementedError(f"Alpha needs to be a power of 2, got {alpha}.")

    # delta_f and f_min for first band, derived from base_domain and chirp_mass_min
    delta_f_band = alpha / ceil_to_power_of_2(
        duration_LO(chirp_mass_min, base_domain.f_min)
    )
    # delta_f can't be smaller than base_domain.delta_f
    delta_f_band = max(delta_f_band, base_domain.delta_f)
    f = base_domain.f_min - base_domain.delta_f / 2.0 + delta_f_band / 2.0
    bands = []

    while f + delta_f_band / 2 < base_domain.f_max:

        f_min_band = f
        while is_within_band(
            f + delta_f_band,
            chirp_mass_min,
            delta_f_band,
            base_domain.f_max,
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
    x = x / np.max(np.abs(x))
    return np.sum((x[..., :-1] * x[..., 1:]) < 0, axis=-1)


def get_period_for_complex_oscillation(
    x: np.ndarray, upper_bound_for_monotonicity: bool = False
):
    """
    Takes complex 1D or 2D array x as input. Returns array of the same shape,
    specifying the cycle length for each bin (axis=-1). This is done by looking at the
    local rate of change of the normalized array x / np.abs(x). Assuming sine-like
    osscillations, the period is related to the maximum rate of change via

        period = 2 pi / max_local(rate_of_change_per_bin).

    Note: this assumes a monotonically increasing period.

    Parameters
    ----------
    x: np.ndarray
        Complex array with oscillation signal. 1D or 2D, oscillation pattern on axis -1.
    upper_bound_for_monotonicity: bool = False
        If set, then the periods returned increase monotonically.

    Returns
    -------
    periods_expanded: np.ndarray
        Array with same shape as x, containing the period (as float) for each bin.
    """
    if not np.iscomplexobj(x):
        raise ValueError("This is only implemented for complex oscillations.")
    if not len(x.shape) in [1, 2]:
        raise ValueError(
            f"Expected shape (num_bins) or (num_waveforms, num_bins), got {x.shape}."
        )
    # Infer period from derivative
    y = x * 1
    if np.min(np.abs(x)) == 0:
        raise ValueError("This function requires |x| > 0.")
    # normalize x
    x = x / np.abs(x)
    # normalized derivative
    dx = np.concatenate(
        ((x[..., 1:] - x[..., :-1]).real, (x[..., 1:] - x[..., :-1]).imag)
    )
    # Infer period from the derivative, assuming sine-like oscillations.
    periods = 2 * np.pi / np.abs(dx)
    if upper_bound_for_monotonicity:
        periods = np.minimum.accumulate((periods)[..., ::-1], axis=-1)[..., ::-1]
    return periods


def get_periods(x: np.ndarray, upper_bound_for_monotonicity: bool = False):
    """
    Takes 1D or 2D array x as input. Returns array of the same shape, specifying the
    cycle length for each bin (axis=-1). This is done by checking for zero-crossings
    in x. The lower/upper boundaries are filled with the periods from the neighboring
    intervals.

    Note: This assumes an oscillatory behavior of x about 0.

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
        # multiply with 2, as a period includes 2 zero crossings
        periods_expanded *= 2
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


def get_band_nodes_for_adaptive_decimation(
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
    initial_downsampling: int
        Downsampling factor of band 0.
    band_nodes: list[int]
        List with nodes for bands.
        Band j consists of indices [nodes[j]:nodes[j+1].
    """
    if len(max_dec_factor_array.shape) != 1:
        raise ValueError("max_dec_factor_array needs to be 1D array.")
    if not (max_dec_factor_array[1:] >= max_dec_factor_array[:-1]).all():
        raise ValueError("max_dec_factor_array needs to increase monotonically.")

    max_dec_factor_array = np.clip(max_dec_factor_array, None, max_dec_factor_global)
    N = len(max_dec_factor_array)
    dec_factor = int(max(1, floor_to_power_of_2(max_dec_factor_array[0])))
    band_nodes = [0]
    upper = dec_factor
    initial_downsampling = dec_factor
    while upper - 1 < N:
        if upper - 1 + dec_factor >= N:
            # conclude while loop, append upper as last node
            band_nodes.append(upper)
        elif dec_factor * 2 <= max_dec_factor_array[upper]:
            # conclude previous band
            band_nodes.append(upper)
            # enter new band
            dec_factor *= 2
        upper += dec_factor

    return initial_downsampling, band_nodes
