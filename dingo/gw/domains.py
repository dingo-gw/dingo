from typing import Dict

import numpy as np
from functools import lru_cache
from abc import ABC, abstractmethod
from dingo.gw.gwutils import *


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

    @abstractmethod
    def time_translate_data(self, data, dt) -> np.ndarray:
        """Time translate strain data by dt seconds."""
        pass

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
    window_kwargs specify windowing used for FFT to obtain FD data from TD
    data in practice.
    """
    domain_type = "uFD"

    def __init__(self, f_min: float, f_max: float, delta_f: float,
                 window_factor: float = None):
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

    def clear_cache_for_all_instances(self):
        """
        Whenever self._f_min and self._f_max are modified, this method needs to
        be the called to clear the cached properties such as
        self.sample_frequencies.

        This clears the cache for the corresponding properties for *all*
        class instances.
        """
        UniformFrequencyDomain.sample_frequencies.fget.cache_clear()
        UniformFrequencyDomain.sample_frequencies_truncated.fget.cache_clear()
        UniformFrequencyDomain.frequency_mask.fget.cache_clear()
        UniformFrequencyDomain.noise_std.fget.cache_clear()

    def set_new_range(self, f_min: float = None, f_max: float = None):
        """
        Set a new range for the domain. This changes the range of the domain to
        [0, f_max], and the truncation range to [f_min, f_max].
        """
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError('f_min must not be larger than f_max.')
        if f_min is not None:
            if self._f_min <= f_min <= self._f_max:
                self._f_min = f_min
            else:
                raise ValueError(f'f_min = {f_min} is not in expected range '
                                 f'[{self._f_min,self._f_max}].')
        if f_max is not None:
            if self._f_min <= f_max <= self._f_max:
                self._f_max = f_max
            else:
                raise ValueError(f'f_max = {f_max} is not in expected range '
                                 f'[{self._f_min, self._f_max}].')
        # clear cached properties, such that they are recomputed when needed
        # instead of using the old (incorrect) ones.
        self.clear_cache_for_all_instances()

    def truncate_data(self, data, allow_for_flexible_upper_bound = False):
        """Truncate data from to [self._f_min, self._f_max]. By convention,
        the last axis is the frequency axis.

        By default, the input data is in the range [0, self._f_max] before
        truncation. In some use cases, the input data has a different range,
        [0, f_max], where f_max > self._f_max. To truncate such data,
        set allow_for_flexible_upper_bound = True.
        """
        if allow_for_flexible_upper_bound:
            return data[...,self.f_min_idx:self.f_max_idx+1]
        else:
            if data.shape[-1] != len(self):
                raise ValueError(f'Expected {len(self)} bins in frequency axis -1, '
                                 f'but got {data.shape[-1]}.')
            return data[...,self.f_min_idx:]

    def time_translate_data(self, data, dt):
        """Time translate complex frequency domain data by dt [in seconds]."""
        if not isinstance(data, np.ndarray):
            raise NotImplementedError(
                f'Only implemented for numpy arrays, got {type(data)}.')
        if not np.iscomplexobj(data):
            raise ValueError(
                'Method expects complex frequency domain data, got real array.')
        # find out whether data is truncated or not
        if data.shape[-1] == len(self):
            f = self.__call__()
        elif data.shape[-1] == self.len_truncated:
            f = self.sample_frequencies_truncated
        else:
            raise ValueError(f'Expected {len(self)} or {self.len_truncated} '
                             f'bins in frequency axis -1, but got '
                             f'{data.shape[-1]}.')
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

    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self._f_max / self._delta_f) + 1

    def __call__(self) -> np.ndarray:
        """Array of uniform frequency bins in the domain [0, f_max]"""
        return self.sample_frequencies

    def __getitem__(self, idx):
        """Slice of uniform frequency grid."""
        sample_frequencies = self.__call__()
        return sample_frequencies[idx]

    @property
    @lru_cache()
    def sample_frequencies(self):
        # print('Computing sample_frequencies.') # To understand caching
        num_bins = self.__len__()
        return np.linspace(0.0, self._f_max, num=num_bins,
                           endpoint=True, dtype=np.float32)

    @property
    @lru_cache()
    def frequency_mask(self) -> np.ndarray:
        """Mask which selects frequency bins greater than or equal to the
        starting frequency"""
        return self.sample_frequencies >= self._f_min

    @property
    def frequency_mask_length(self) -> int:
        """Number of samples in the subdomain domain[frequency_mask]."""
        mask = self.frequency_mask
        return len(np.flatnonzero(np.asarray(mask)))

    @property
    def f_min_idx(self):
        return round(self._f_min / self._delta_f)

    @property
    def f_max_idx(self):
        return round(self._f_max / self._delta_f)

    @property
    @lru_cache()
    def sample_frequencies_truncated(self):
        return self.sample_frequencies[self.f_min_idx:]

    @property
    def len_truncated(self):
        return len(self.sample_frequencies_truncated)

    @property
    def window_factor(self):
        return self._window_factor

    @window_factor.setter
    def window_factor(self, value):
        """Set self._window_factor and clear cache of self.noise_std."""
        self._window_factor = value
        UniformFrequencyDomain.noise_std.fget.cache_clear()

    @property
    @lru_cache()
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
            raise ValueError('Window factor needs to be set for noise_std.')
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._f_max

    @property
    def f_min(self) -> float:
        """The minimum frequency [Hz]."""
        return self._f_min

    @property
    def delta_f(self) -> float:
        """The frequency spacing of the uniform grid [Hz]."""
        return self._delta_f

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

    def time_translate_data(self, data, dt) -> np.ndarray:
        raise NotImplementedError


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
    if set(domain_settings.keys()) != {'name', 'kwargs'}:
        raise ValueError(f'Got domain_settings {domain_settings.keys()}, '
                         f'expected dict_keys([\'name\', \'kwargs\'])')
    if domain_settings['name'] == 'UniformFrequencyDomain':
        return UniformFrequencyDomain(**domain_settings['kwargs'])
    elif domain_settings['name'] == 'TimeDomain':
        return TimeDomain(**domain_settings['kwargs'])
    else:
        raise ValueError(f'Domain {domain_settings["name"]} not implemented.')


if __name__ == '__main__':
    kwargs = {'f_min': 20, 'f_max': 2048, 'delta_f': 0.125}
    domain = UniformFrequencyDomain(**kwargs)

    d1 = domain()
    d2 = domain()
    print('Clearing cache.', end=' ')
    domain.clear_cache_for_all_instances()
    print('Done.')
    d3 = domain()

    print('Changing domain range.', end=' ')
    domain.set_new_range(20, 100)
    print('Done.')

    d4 = domain()
    d5 = domain()

    print(len(d1), len(d4))