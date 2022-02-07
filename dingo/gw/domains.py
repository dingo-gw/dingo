from typing import Dict

from functools import lru_cache
from abc import ABC, abstractmethod
from dingo.gw.gwutils import *


class Domain(ABC):
    """Defines the physical domain on which the data of interest live.

    This includes a specification of the bins or points,
    and a few additional properties associated with the data.
    """

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
    @abstractmethod
    def sampling_rate(self) -> float:
        """The sampling rate of the data [Hz]."""
        pass

    @property
    @abstractmethod
    def f_max(self) -> float:
        """The maximum frequency [Hz] is set to half the sampling rate."""

    @property
    @abstractmethod
    def duration(self) -> float:
        """Waveform duration in seconds."""
        pass

    @property
    @abstractmethod
    def min_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def max_idx(self) -> int:
        pass

    @property
    @abstractmethod
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        pass


class FrequencyDomain(Domain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max]
    with spacing delta_f.
    Given a finite length of time domain data, the Fourier domain data
    starts at a frequency f_min and is zero below this frequency.
    window_kwargs specify windowing used for FFT to obtain FD data from TD
    data in practice.
    """

    def __init__(
        self, f_min: float, f_max: float, delta_f: float, window_factor: float = None
    ):
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

    @staticmethod
    def clear_cache_for_all_instances():
        """
        Whenever self._f_min and self._f_max are modified, this method needs to
        be the called to clear the cached properties such as
        self.sample_frequencies.

        This clears the cache for the corresponding properties for *all*
        class instances.
        """
        FrequencyDomain.sample_frequencies.fget.cache_clear()
        FrequencyDomain.sample_frequencies_truncated.fget.cache_clear()
        FrequencyDomain.frequency_mask.fget.cache_clear()
        FrequencyDomain.noise_std.fget.cache_clear()

    def update(self, new_settings):
        new_settings = new_settings.copy()
        if "type" in new_settings and new_settings.pop("type") not in [
            "FrequencyDomain",
            "FD",
        ]:
            raise ValueError("Cannot update domain to type other than FrequencyDomain.")
        for k, v in new_settings.items():
            if k not in ['f_min', 'f_max', 'delta_f', 'window_factor']:
                raise KeyError(f'Invalid key for domain update: {k}.')
            if k == 'window_factor' and v != self._window_factor:
                raise ValueError('Cannot update window_factor.')
            if k == 'delta_f' and v != self._delta_f:
                raise ValueError('Cannot update delta_f.')
        self.set_new_range(f_min=new_settings.get('f_min', None),
                           f_max=new_settings.get('f_max', None))

    def set_new_range(self, f_min: float = None, f_max: float = None):
        """
        Set a new range for the domain. This changes the range of the domain to
        [0, f_max], and the truncation range to [f_min, f_max].
        """
        if f_min is not None and f_max is not None and f_min >= f_max:
            raise ValueError("f_min must not be larger than f_max.")
        if f_min is not None:
            if self._f_min <= f_min <= self._f_max:
                self._f_min = f_min
            else:
                raise ValueError(
                    f"f_min = {f_min} is not in expected range "
                    f"[{self._f_min,self._f_max}]."
                )
        if f_max is not None:
            if self._f_min <= f_max <= self._f_max:
                self._f_max = f_max
            else:
                raise ValueError(
                    f"f_max = {f_max} is not in expected range "
                    f"[{self._f_min, self._f_max}]."
                )
        # clear cached properties, such that they are recomputed when needed
        # instead of using the old (incorrect) ones.
        self.clear_cache_for_all_instances()

    def adjust_data_range(self, data, axis=-1, low_value=0.0):
        sl = [slice(None)] * data.ndim

        # First truncate beyond f_max.
        sl[axis] = slice(0, self.max_idx + 1)
        data = data[tuple(sl)]

        # Set data value below f_min to low_value.
        sl[axis] = slice(0, self.min_idx)
        data[tuple(sl)] = low_value

        return data

    def truncate_data(self, data, axis=-1, allow_for_flexible_upper_bound=False):
        """Truncate data from to [self._f_min, self._f_max]. By convention,
        the last axis is the frequency axis.

        By default, the input data is in the range [0, self._f_max] before
        truncation. In some use cases, the input data has a different range,
        [0, f_max], where f_max > self._f_max. To truncate such data,
        set allow_for_flexible_upper_bound = True.
        """
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(self.min_idx, self.max_idx + 1)
        return data[tuple(sl)]

        # Why do we need separate cases here? I believe I unified them above.
        # I also removed a test that tests for this special case.

        # if allow_for_flexible_upper_bound:
        #     return data[...,self.f_min_idx:self.f_max_idx+1]
        # else:
        #     if data.shape[-1] != len(self):
        #         raise ValueError(f'Expected {len(self)} bins in frequency axis -1, '
        #                          f'but got {data.shape[-1]}.')
        #     return data[...,self.f_min_idx:]

    def time_translate_data(self, data, dt):
        """Time translate complex frequency domain data by dt [in seconds]."""
        if not isinstance(data, np.ndarray):
            raise NotImplementedError(
                f"Only implemented for numpy arrays, got {type(data)}."
            )
        if not np.iscomplexobj(data):
            raise ValueError(
                "Method expects complex frequency domain data, got real array."
            )
        # find out whether data is truncated or not
        if data.shape[-1] == len(self):
            f = self.__call__()
        elif data.shape[-1] == self.len_truncated:
            f = self.sample_frequencies_truncated
        else:
            raise ValueError(
                f"Expected {len(self)} or {self.len_truncated} "
                f"bins in frequency axis -1, but got "
                f"{data.shape[-1]}."
            )
        # shift data
        return data * np.exp(-2j * np.pi * dt * f)

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
        return np.linspace(
            0.0, self._f_max, num=num_bins, endpoint=True, dtype=np.float32
        )

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
    def min_idx(self):
        return round(self._f_min / self._delta_f)

    @property
    def max_idx(self):
        return round(self._f_max / self._delta_f)

    @property
    @lru_cache()
    def sample_frequencies_truncated(self):
        return self.sample_frequencies[self.min_idx:]

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
        FrequencyDomain.noise_std.fget.cache_clear()

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
            raise ValueError("Window factor needs to be set for noise_std.")
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
    def sampling_rate(self) -> float:
        return 2.0 * self._f_max

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "FrequencyDomain",
            "f_min": self._f_min,
            "f_max": self._f_max,
            "delta_f": self._delta_f,
            "window_factor": self._window_factor,
        }


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
    """

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
        return np.linspace(
            0.0, self._time_duration, num=num_bins, endpoint=False, dtype=np.float32
        )

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

    @property
    def f_max(self) -> float:
        """The maximum frequency [Hz] is typically set to half the sampling
        rate."""
        return self._sampling_rate / 2.0

    @property
    def duration(self) -> float:
        """Waveform duration in seconds."""
        return self._time_duration

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def min_idx(self) -> int:
        return 0

    @property
    def max_idx(self) -> int:
        return round(self._time_duration * self._sampling_rate)

    @property
    def domain_dict(self):
        """Enables to rebuild the domain via calling build_domain(domain_dict)."""
        return {
            "type": "TimeDomain",
            "time_duration": self._time_duration,
            "sampling_rate": self._sampling_rate,
        }


class PCADomain(Domain):
    """TODO"""

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


def build_domain(settings: Dict) -> Domain:
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    settings : dict
        Dicionary with 'type' key denoting the type of domain, and keys corresponding
        to the kwargs needed to construct the Domain.

    Returns
    -------
    A Domain instance of the correct type.
    """
    if "type" not in settings:
        raise ValueError(
            f'Domain settings must include a "type" key. Settings included '
            f"the keys {settings.keys()}."
        )

    # The settings other than 'type' correspond to the kwargs of the Domain constructor.
    kwargs = {k: v for k, v in settings.items() if k != "type"}
    if settings["type"] in ["FrequencyDomain", "FD"]:
        return FrequencyDomain(**kwargs)
    elif settings["type"] == ["TimeDomain", "TD"]:
        return TimeDomain(**kwargs)
    else:
        raise NotImplementedError(f'Domain {settings["name"]} not implemented.')


if __name__ == "__main__":
    kwargs = {"f_min": 20, "f_max": 2048, "delta_f": 0.125}
    domain = FrequencyDomain(**kwargs)

    d1 = domain()
    d2 = domain()
    print("Clearing cache.", end=" ")
    domain.clear_cache_for_all_instances()
    print("Done.")
    d3 = domain()

    print("Changing domain range.", end=" ")
    domain.set_new_range(20, 100)
    print("Done.")

    d4 = domain()
    d5 = domain()

    print(len(d1), len(d4))
