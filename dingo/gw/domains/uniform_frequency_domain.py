from typing import Optional

import numpy as np
import torch

from dingo.gw.gwutils import *
from .base_frequency_domain import BaseFrequencyDomain


class UniformFrequencyDomain(BaseFrequencyDomain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max] with spacing
    delta_f.

    Given a finite length of time domain data, the Fourier domain data starts
    at a frequency f_min and is zero below this frequency.

    window_kwargs specify windowing used for FFT to obtain FD data from TD data in
    practice.
    """

    def __init__(
        self,
        f_min: float,
        f_max: float,
        delta_f: float,
        window_factor: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        f_min : float
        f_max : float
        delta_f : float
        window_factor Optional[float]
        """
        super().__init__()
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor
        self._frequency_mask = None

    def update(self, new_settings: dict):
        """
        Update the domain with new settings. This is only allowed if the new settings
        are "compatible" with the old ones. E.g., f_min should be larger than the
        existing f_min.

        Parameters
        ----------
        new_settings : dict
            Settings dictionary. Must contain a subset of the keys contained in
            domain_dict.
        """
        new_settings = new_settings.copy()
        if "type" in new_settings and new_settings.pop("type") not in [
            "UniformFrequencyDomain",
            "FrequencyDomain",
            "FD",
        ]:
            raise ValueError(
                "Cannot update domain to type other than UniformFrequencyDomain."
            )
        for k, v in new_settings.items():
            if k not in ["f_min", "f_max", "delta_f", "window_factor"]:
                raise KeyError(f"Invalid key for domain update: {k}.")
            if k == "window_factor" and v != self._window_factor:
                raise ValueError("Cannot update window_factor.")
            if k == "delta_f" and v != self._delta_f:
                raise ValueError("Cannot update delta_f.")
        self._set_new_range(
            f_min=new_settings.get("f_min", None), f_max=new_settings.get("f_max", None)
        )

    def _set_new_range(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ) -> None:
        """
        Set a new [f_min, f_max] range for the domain. Both endpoints must be in the
        existing sample_frequencies, and f_min < f_max. Neither endpoint may lie
        outside the current range.
        """
        new_min = f_min if f_min is not None else self.f_min
        new_max = f_max if f_max is not None else self.f_max

        if new_min >= new_max:
            raise ValueError("f_min must be strictly less than f_max.")

        if not (self.f_min <= new_min and new_max <= self.f_max):
            raise ValueError(
                f"Requested range [{new_min}, {new_max}] lies outside "
                f"original range [{self.f_min}, {self.f_max}]."
            )

        missing = [x for x in (new_min, new_max) if x not in self.sample_frequencies]
        if missing:
            raise ValueError(f"Endpoints {missing} not in existing sample_frequencies.")

        self.f_min, self.f_max = new_min, new_max

    def update_data(self, data: np.ndarray, axis: int = -1, low_value: float = 0.0):
        """
        Adjusts data to be compatible with the domain:

        * Below f_min, it sets the data to low_value (typically 0.0 for a waveform,
          but for a PSD this might be a large value).
        * Above f_max, it truncates the data array.

        Parameters
        ----------
        data : np.ndarray
            Data array
        axis : int
            Which data axis to apply the adjustment along.
        low_value : float
            Below f_min, set the data to this value.

        Returns
        -------
        np.ndarray
            The new data array.
        """
        sl = [slice(None)] * data.ndim

        # First truncate beyond f_max.
        sl[axis] = slice(0, self.max_idx + 1)
        data = data[tuple(sl)]

        # Set data value below f_min to low_value.
        sl[axis] = slice(0, self.min_idx)
        data[tuple(sl)] = low_value

        return data

    def get_sample_frequencies_astype(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Returns a 1D frequency array compatible with the last index of data array.

        Decides whether array is numpy or torch tensor (and cuda vs cpu), and whether it
        contains the leading zeros below f_min.

        Parameters
        ----------
        data : Union[np.array, torch.Tensor]
            Sample data

        Returns
        -------
        frequency array compatible with last index
        """
        f = super().get_sample_frequencies_astype(data)

        # Whether to include zeros below f_min
        if data.shape[-1] == len(self) - self.min_idx:
            f = f[self.min_idx :]
        elif data.shape[-1] != len(self):
            raise TypeError(
                f"Data with {data.shape[-1]} frequency bins is "
                f"incompatible with domain."
            )

        return f

    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self.f_max / self.delta_f) + 1

    @property
    def sample_frequencies(self) -> np.ndarray:
        if self._sample_frequencies is None:
            num_bins = len(self)
            self._sample_frequencies = np.linspace(
                0.0, self.f_max, num=num_bins, endpoint=True, dtype=np.float32
            )
        return self._sample_frequencies

    @property
    def frequency_mask(self) -> np.ndarray:
        """Mask which selects frequency bins greater than or equal to the
        starting frequency"""
        if self._frequency_mask is None:
            self._frequency_mask = self.sample_frequencies >= self.f_min
        return self._frequency_mask

    def _reset_caches(self):
        self._sample_frequencies = None
        self._sample_frequencies_torch = None
        self._sample_frequencies_torch_cuda = None
        self._frequency_mask = None

    @property
    def frequency_mask_length(self) -> int:
        """Number of samples in the subdomain domain[frequency_mask]."""
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

    @property
    def min_idx(self) -> int:
        return round(self._f_min / self._delta_f)

    @property
    def max_idx(self) -> int:
        return round(self._f_max / self._delta_f)

    @property
    def window_factor(self) -> float:
        return self._window_factor

    @window_factor.setter
    def window_factor(self, value: float):
        """Set self._window_factor."""
        self._window_factor = float(value)

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._f_max

    @f_max.setter
    def f_max(self, value):
        self._f_max = float(value)
        self._reset_caches()

    @property
    def f_min(self) -> float:
        """The minimum frequency [Hz]."""
        return self._f_min

    @f_min.setter
    def f_min(self, value):
        self._f_min = float(value)
        self._reset_caches()

    @property
    def delta_f(self) -> float:
        """The frequency spacing of the uniform grid [Hz]."""
        return self._delta_f

    @delta_f.setter
    def delta_f(self, value):
        self._delta_f = float(value)
        self._reset_caches()

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return 1.0 / self.delta_f

    @property
    def sampling_rate(self) -> float:
        return 2.0 * self.f_max

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "UniformFrequencyDomain",
            "f_min": self.f_min,
            "f_max": self.f_max,
            "delta_f": self.delta_f,
            "window_factor": self.window_factor,
        }
