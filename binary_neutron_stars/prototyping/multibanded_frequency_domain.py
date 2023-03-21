from typing import Iterable, Union
import numpy as np
import torch

from dingo.gw.domains import Domain, FrequencyDomain, build_domain
from multibanding_utils import decimate_uniform


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
        bands: Iterable[Iterable[float]],
        original_domain: Union[FrequencyDomain, dict] = None,
    ):
        """
        Parameters
        ----------
        bands: Iterable[Iterable[float]]
            Frequency bands. Each band contains three elements, [f_min, f_max, delta_f].
        original_domain: Union[FrequencyDomain, dict] = None
            Original (uniform frequency) domain of data, which is the starting point
            for the decimation. This determines the decimation details and the noise_std.
            Either provided as dict for build_domain, or as domain_object.
        """
        self._bands = np.array(bands)
        self.initialize_bands()
        if type(original_domain) == dict:
            original_domain = build_domain(original_domain)
        self.original_domain = original_domain
        if self.original_domain is not None:
            self.initialize_decimation()

    @classmethod
    def init_for_decimation(
        cls, original_domain, chirp_mass_min, alpha_bands, delta_f_max
    ):
        bands = get_bands_for_decimation(
            original_domain, chirp_mass_min, alpha_bands, delta_f_max
        )
        return cls(bands, original_domain)

    def initialize_bands(self):
        if len(self._bands.shape) != 2 or self._bands.shape[1] != 3:
            raise ValueError(
                f"Expected format [num_bands, 3] for bands, got {self._bands.shape}."
            )
        self.num_bands = len(self._bands)
        self._f_min_bands = self._bands[:, 0]
        self._f_max_bands = self._bands[:, 1]
        self._delta_f_bands = self._bands[:, 2]
        self._num_bins_bands = np.array(
            [int((band[1] - band[0]) / band[2] + 1) for band in self._bands]
        )
        self._f_bands = [
            np.linspace(f_min, f_max, num_bins)
            for f_min, f_max, num_bins in zip(
                self._f_min_bands, self._f_max_bands, self._num_bins_bands
            )
        ]
        for f_band, delta_f_band in zip(self._f_bands, self._delta_f_bands):
            if len(f_band) > 1:
                assert f_band[1] - f_band[0] == delta_f_band
        self._sample_frequencies = np.concatenate(self._f_bands)
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None
        self._f_min = self._sample_frequencies[0]
        self._f_max = self._sample_frequencies[-1]

    def initialize_decimation(self):
        if self.original_domain is None:
            raise ValueError(
                "Original domain needs to be specified to initialize decimation."
            )
        if not np.all(self._delta_f_bands % self.original_domain.delta_f == 0):
            raise NotImplementedError(
                "delta_f_bands need to be multiple of delta_f of the original domain."
            )
        self.decimation_inds_bands = []
        for f_min_band, f_max_band, delta_f_band in self._bands:
            decimation_factor_band = int(delta_f_band / self.original_domain.delta_f)
            idx_lower_band = int(
                (f_min_band - delta_f_band / 2.0 + self.original_domain.delta_f / 2.0)
                / self.original_domain.delta_f
            )
            # idx_upper_band is *inclusive*, so one slices with
            # [...idx_lower_band:idx_upper_band + 1]
            idx_upper_band = int(
                (f_max_band + delta_f_band / 2.0 - self.original_domain.delta_f / 2.0)
                / self.original_domain.delta_f
            )
            self.decimation_inds_bands.append(
                [idx_lower_band, idx_upper_band, decimation_factor_band]
            )

    def decimate(self, data):
        if data.shape[-1] == len(self.original_domain):
            offset_idx = 0
        elif data.shape[-1] == len(self.original_domain) - self.original_domain.min_idx:
            offset_idx = -self.original_domain.min_idx
        else:
            raise ValueError(
                f"Provided data has {data.shape[-1]} bins, which is incompatible with "
                f"the expected domain of length {len(self.original_domain)}"
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
        for idx, (lower_in, upper_in, dec_factor) in enumerate(
            self.decimation_inds_bands
        ):
            # boundaries in original data
            lower_in += offset_idx
            upper_in += offset_idx
            num_bins = self._num_bins_bands[idx]
            data_decimated[..., lower_out : lower_out + num_bins] = decimate_uniform(
                data[..., lower_in : upper_in + 1], dec_factor
            )
            lower_out += num_bins
        return data_decimated

    def update(self, new_settings: dict):
        raise NotImplementedError()

    def set_new_range(self, f_min: float = None, f_max: float = None):
        raise NotImplementedError()

    def update_data(self, data: np.ndarray, axis: int = -1, low_value: float = 0.0):
        raise NotImplementedError()

    def time_translate_data(self, data, dt):
        """
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    @window_factor.setter
    def window_factor(self, value):
        """Set self._window_factor and clear cache of self.noise_std."""
        raise NotImplementedError()

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        TODO: This description makes some assumptions that need to be clarified.
        TODO: Adapt to multibanded domain
        Windowing of TD data; tapering window has a slope -> reduces power only for noise,
        but not for the signal which is in the main part unaffected by the taper
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
        # return {
        #     "type": "MultibandedFrequencyDomain",
        #     "f_min": self.f_min,
        #     "f_max": self.f_max,
        #     "delta_f": self.delta_f,
        #     "window_factor": self.window_factor,
        # }
