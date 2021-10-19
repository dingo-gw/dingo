from typing import Dict

import numpy as np
from functools import lru_cache
from abc import ABC, abstractmethod
from .gwutils import *


class Domain(ABC):
    """Defines the physical domain on which the data of interest live.

    This includes a specification of the bins or points,
    and a few additional properties associated with the data.
    """
    domain_type: str

    @abstractmethod
    def __len__(self):
        """Number of bins or points in the domain"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Array of bins in the domain"""
        pass

    # @abstractmethod
    # def time_translate_strain_data(self, strain_data, dt) -> np.ndarray:
    #     """Time translate strain data by dt seconds."""
    #     pass

    @property
    @abstractmethod
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution"""
        # FIXME: For this to make sense, it assumes knowledge about how the domain is used in conjunction
        #  with (waveform) data, whitening and adding noise. Is this the best place to define this?
        pass

    @property
    def sampling_rate(self) -> float:
        """The sampling rate of the data [Hz]."""
        pass

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is set to half the sampling rate."""

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        pass

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict).
        """
        pass


class UniformFrequencyDomain(Domain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max]
    with spacing delta_f.
    Given a finite length of time domain data, the Fourier domain data
    starts at a frequency f_min and is zero below this frequency.
    window_factor is used to compute noise_std().
    """
    domain_type = "uFD"

    def __init__(self, f_min: float, f_max: float, delta_f: float,
                 window_factor: float, truncation_range: tuple = None):
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor
        self.initialize_truncation(truncation_range)

    def initialize_truncation(self, truncation_range):
        """Initializes truncation with truncation_range."""
        self._truncation_range = truncation_range
        if self._truncation_range is not None:
            f_lower, f_upper = self._truncation_range
            if not self._f_min <= f_lower < f_upper <= self._f_max:
                raise ValueError(
                    f'Invalid truncation range [{f_lower},{f_upper}] for '
                    f'frequency range [{self._f_min},{self._f_max}].')
            self._truncation_idx_lower = round(f_lower/self._delta_f)
            self._truncation_idx_upper = round(f_upper/self._delta_f) + 1
            self._truncation_num_bins = self._truncation_idx_upper - \
                                        self._truncation_idx_lower
            self._truncated_sample_frequencies = \
                self[self._truncation_idx_lower:self._truncation_idx_upper]

    def truncate_data(self, data, axis=None):
        """Truncate data to self._truncation_range. If axis is not specified
        it is detected automatically."""
        if self._truncation_range is None:
            raise ValueError('Truncation not initialized. Call '
                             'self.initialize_truncation with appropriate '
                             'range.')
        if axis is None:
            axis = find_axis(data, len(self))
        if data.shape[axis] != len(self):
            raise ValueError(f'Truncation along axis {axis} failed. Dim '
                             f'{data.shape[axis]} instead of {len(self)}.')
        return truncate_array(data, axis, self._truncation_idx_lower,
                              self._truncation_idx_upper)

    def time_translate_data(self, data, dt):
        """Time translate complex frequency domain data by dt [in seconds]."""
        if not isinstance(data, np.ndarray):
            raise NotImplementedError(f'Only implemented for numpy arrays, '
                                      f'got {type(data)}.')
        if not np.iscomplexobj(data):
            raise ValueError('Method expects complex frequency domain data, '
                             'got real array.')
        # find out whether data is truncated or not
        ax0 = np.where(np.array(data.shape) == len(self))[0]
        if self._truncation_range is not None:
            ax1 = np.where(np.array(data.shape) == self._truncation_num_bins)[0]
        if len(ax0) + len(ax1) != 1:
            raise NotImplementedError('Can not identify unique frequency axis.')
        elif len(ax0) == 1:
            f = self.__call__()
        else:
            f = self._truncated_sample_frequencies
        # shift data
        return data * np.exp(- 2j * np.pi * dt * f)

    # def time_translate_batch(self, data, dt, axis=None):
    #     # h_d * np.exp(- 2j * np.pi * time_shift * self.sample_frequencies)
    #     if isinstance(data, np.ndarray):
    #         if np.iscomplexobj(data):
    #             pass
    #         else:
    #             pass
    #     elif isinstance(data, torch.Tensor):
    #         pass
    #     else:
    #         raise NotImplementedError(f'Method only implemented for np arrays '
    #                                   f'and torch tensors, got {type(data)}')


    @lru_cache()
    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self._f_max / self._delta_f) + 1

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """Array of uniform frequency bins in the domain [0, f_max]"""
        num_bins = self.__len__()
        sample_frequencies = np.linspace(0.0, self._f_max, num=num_bins, endpoint=True, dtype=np.float32)
        return sample_frequencies

    def __getitem__(self, idx):
        """Slice of uniform frequency grid."""
        sample_frequencies = self.__call__()
        return sample_frequencies[idx]

    @property
    @lru_cache()
    def frequency_mask(self) -> np.ndarray:
        """Mask which selects frequency bins greater than or equal to the starting frequency"""
        sample_frequencies = self.__call__()
        return sample_frequencies >= self._f_min

    @property
    def frequency_mask_length(self) -> int:
        """Number of samples in the subdomain domain[frequency_mask]."""
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

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
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)


    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is set to half the sampling rate."""
        return self._f_max

    @f_max.setter
    def f_max(self, f_max: float):
        self._f_max = f_max

    @property
    def f_min(self) -> float:
        """The minimum frequency [Hz]."""
        return self._f_min

    @property
    def delta_f(self) -> float:
        """The frequency spacing of the uniform grid [Hz]."""
        return self._delta_f

    @property
    def sampling_rate(self) -> float:
        """The sampling rate of the data [Hz]."""
        return 2.0 * self._f_max

    @sampling_rate.setter
    def sampling_rate(self, fs: float):
        self._f_max = fs / 2.0

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return 1.0 / self._delta_f

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict).
        """
        kwargs = {'f_min': self._f_min,
                  'f_max': self._f_max,
                  'delta_f': self._delta_f,
                  'window_factor': self._window_factor,
                  'truncation_range': self._truncation_range,
                  }
        return {'name': 'UniformFrequencyDomain', 'kwargs': kwargs}


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
    """
    domain_type = "TD"

    def __init__(self, time_duration: float, sampling_rate: float):
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    @lru_cache()
    def __len__(self):
        """Number of time bins given duration and sampling rate"""
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self) -> np.ndarray:
        """Array of uniform times at which data is sampled"""
        num_bins = self.__len__()
        return np.linspace(0.0, self._time_duration, num=num_bins,
                           endpoint=False, dtype=np.float32)

    @property
    def delta_t(self) -> float:
        """The size of the time bins"""
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t: float):
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)


class PCADomain(Domain):
    """TODO"""
    domain_type = "PCA"

    # Not super important right now
    # FIXME: Should this be defined for FD or TD bases or both?
    # Nrb instead of Nf

    @property
    def noise_std(self) -> float:
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        # FIXME
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)


class NonuniformFrequencyDomain(Domain):
    """TODO"""
    domain_type = "nFD"

    # It probably doesn't make sense to inherit from FrequencyDomain; we'll need this for low mass binaries
    pass


def build_domain(domain_settings: Dict):
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    domain_settings:
        A dictionary of settings for the domain class.
    """
    if domain_settings['name'] == 'UniformFrequencyDomain':
        return UniformFrequencyDomain(**domain_settings['kwargs'])
    elif domain_settings['name'] == 'TimeDomain':
        return TimeDomain(**domain_settings['kwargs'])
    else:
        raise ValueError(f'Domain {domain_settings["name"]} not implemented.')