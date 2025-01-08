from typing import Iterable, Union

import numpy as np
import torch

from .base import Domain
from .frequency_domain import FrequencyDomain


class IrregularFrequencyDomain(Domain):
    """
    Defines the physical domain of the data of interest.

    The frequency bins do not have to follow an specific spacing.
    but should be montonically increasing.
    The data below a frequency of f_min is set to 0.

    TODO change the current FrequencyDomain class into a class
    "UniformFrequencyDomain". Then IrregularFrequencyDomain,
    MultiBandedFrequencyDomain and UniformFrequency domain should
    all inherit from this

    """

    def __init__(
        self, sample_frequencies: Iterable[float], window_factor: float = None
    ):
        """
        Parameters
        ----------
        sample_frequencies : Iterable[float]
            Frequency bins of the domain. Can be any arbitrary set of frequencies

        window_factor: float = None
            Window factor for this domain. Required when using self.noise_std.
        """

        if not np.all(np.diff(sample_frequencies) > 0):
            raise ValueError("sample_frequencies should be monotonicall increasing")

        self._sample_frequencies = sample_frequencies
        self.f_min = sample_frequencies[0]
        self.f_max = sample_frequencies[-1]
        self.window_factor = window_factor

    def update(self, new_settings: dict):
        """
        Update the domain by truncating the frequency range.

        After calling this function, data from the initial (unupdated) domain can be
        updated to this domain with self.update(data), which truncates the data
        accordingly. We don't allow for multiple updates of the domain to avoid bugs
        due to an unclear initial domain.

        Parameters
        ----------
        new_settings : dict
            Settings dictionary. Keys must either be a subset of the keys contained in
            domain_dict, or a subset of ["f_min", "f_max"].
        """
        if set(new_settings.keys()).issubset(["f_min", "f_max"]):
            self.set_new_range(**new_settings)
        elif set(new_settings.keys()) == self.domain_dict.keys():
            if new_settings == self.domain_dict:
                return
            self.set_new_range(
                f_min=new_settings["base_domain"]["f_min"],
                f_max=new_settings["base_domain"]["f_max"],
            )
            if self.domain_dict != new_settings:
                raise ValueError(
                    f"Update settings {new_settings} are incompatible with "
                    f"domain settings {self.domain_dict}."
                )
        else:
            raise ValueError(
                f"Invalid argument for domain update {new_settings}. Must either be "
                f'{list(self.domain_dict.keys())} or a subset of ["f_min, f_max"].'
            )

    def set_new_range(self, f_min: float = None, f_max: float = None):
        """
        Set a new range [f_min, f_max] for the domain. This operation is only allowed
        if the new range is contained within the old one.

        Note: f_min, f_max correspond to the range in the *base_domain*.
        """
        if f_min is None and f_max is None:
            return
        if self.range_update_initial_length is not None:
            raise ValueError(f"Can't update domain of type {type(self)} a second time.")
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError("f_min must not be larger than f_max.")

        lower_bin, upper_bin = 0, len(self) - 1

        if f_min is not None:
            if self._f_base_lower[0] <= f_min <= self._f_base_lower[-1]:
                # find new starting bin (first element with f >= f_min)
                lower_bin = np.where(self._f_base_lower >= f_min)[0][0]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[-1], self._f_base_lower[-1]}]."
                )

        if f_max is not None:
            if self._f_base_upper[0] <= f_max <= self._f_base_upper[-1]:
                # find new final bin (last element where f <= f_max)
                upper_bin = np.where(self._f_base_upper <= f_max)[0][-1]
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_base_lower[-1], self._f_base_lower[-1]}]."
                )

        lower_band = self._band_assignment[lower_bin]
        upper_band = self._band_assignment[upper_bin]
        # new nodes extend to upper_band + 2: we have +1 from the exclusive end index
        # and +1, as we have num_bands + 1 elements in nodes
        nodes_new = copy(self.nodes)[lower_band : upper_band + 2]
        nodes_new[0] = self._f_base_lower[lower_bin]
        nodes_new[-1] = self._f_base_upper[upper_bin] + self.base_domain.delta_f

        self.range_update_initial_length = len(self)
        self.range_update_idx_lower = lower_bin
        self.range_update_idx_upper = upper_bin

        self.nodes = nodes_new
        self.initialize_bands(self._delta_f_bands[lower_band])
        assert self.range_update_idx_upper - self.range_update_idx_lower + 1 == len(
            self
        )

    def update_data(
        self,
        data: np.ndarray,
        old_sample_frequencies: np.ndarray = None,
        low_value: float = 0,
        axis: int = -1,
    ):
        """
        Adjusts the data to be compatible with the domain. Given a previous set of sample
        frequencies corresponding to data, will return a new array corresponding to self.sample_frequencies.
        Will raise an error of self.sample_frequencies are not in the old_sample_frequencies
        """
        if old_sample_frequencies is None or data.shape[axis] == len(self):
            return data
        elif not np.all(np.isin(self.sample_frequencies, old_sample_frequencies)):
            raise ValueError(
                "sample_frequencies is not a subset of old_sample_frequencies"
            )
        else:
            sl = np.where(np.isin(old_sample_frequencies, self.sample_frequencies))[0]
            data = data[sl]
            return data

    def time_translate_data(self, data, dt):
        """
        TODO: like self.add_phase, this is just copied from FrequencyDomain and
        TODO: could be inherited instead.
        Time translate frequency-domain data by dt. Time translation corresponds (in
        frequency domain) to multiplication by

        .. math::
            \\exp(-2 \\pi i \\, f \\, dt).

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

    @sample_frequencies.setter
    def sample_frequencies(self, value):
        self._sample_frequencies = value

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

    @property
    def frequency_mask_length(self) -> int:
        raise NotImplementedError()

    @property
    def min_idx(self):
        return 0

    @property
    def max_idx(self):
        raise len(self.sample_frequencies) - 1

    @property
    def window_factor(self):
        return self.window_factor

    @window_factor.setter
    def window_factor(self, value):
        """Set window factor of base domain."""
        self._window_factor = value

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        TODO: This description makes some assumptions that need to be clarified.
        Windowing of TD data; tapering window has a slope -> reduces power only for noise,
        but not for the signal which is in the main part unaffected by the taper
        """
        if self._window_factor is None:
            raise ValueError("Window factor needs to be set for noise_std.")
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    def f_max(self) -> float:
        return self.f_max

    @f_max.setter
    def f_max(self, value):
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        self._f_max = float(value)

    @property
    def f_min(self) -> float:
        return self.f_min

    @f_min.setter
    def f_min(self, value):
        self._f_min = float(value)

    @property
    def delta_f(self) -> float:
        return np.diff(self.sample_frequencies)

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
            "type": "IrregularFrequencyDomain",
            "sample_frequencies": self.sample_frequencies,
            "window_factor": self.window_factor,
        }
