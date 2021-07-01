import numpy as np
from functools import lru_cache
from abc import ABC, abstractmethod


# Note: I made Domain an abstract base class:
#  - i.e. an "interface" which cannot be instantiated by itself
#  - abstract methods need to be overridden in derived classes

# TODO: Should Domains have any other behavior? Where do they interface with other classes?
#  - A Domain must always be coupled with (waveform) data to be useful
#  - classes Waveform, FrequencyDomainWaveform, TimeDomainWaveform should have a Domain attribute

class Domain(ABC):
    """Defines the physical domain on which the data of interest live.

    This includes a specification of the bins or points,
    and a few additional properties associated with the data.
    """

    @abstractmethod
    def __len__(self):
        """Number of bins or points in the domain"""
        # Note: This generalizes Nf, Nt, Nrb
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Array of bins in the domain"""
        # Note: generalizes sample_frequencies or times
        pass

    @property
    @abstractmethod
    def noise_std(self):
        """Standard deviation of the whitened noise distribution"""
        # FIXME: For this to make sense, it assumes knowledge about how the domain is used in conjunction
        #  with (waveform) data, whitening and adding noise. Is this the best place to define this?
        pass

    @abstractmethod
    def context_dim(self, num_detectors):
        """Dimensionality of the strain data in the entire network"""
        # FIXME: * may be better not to include this here and do the calculation manually outside this class
        pass


class UniformFrequencyDomain(Domain):
    """Defines the physical domain on which the data of interest live.

    The frequency bins are assumed to be uniform between [0, f_max]
    with spacing delta_f.
    Given a finite length of time domain data, the Fourier domain data
    starts at a frequency f_min and is zero below this frequency.
    window_factor is used to compute noise_std().
    """

    def __init__(self, f_min, f_max, delta_f, window_factor):
        self._f_min = f_min
        self._f_max = f_max
        self._delta_f = delta_f
        self._window_factor = window_factor

    @lru_cache()
    def __len__(self):
        """Number of frequency bins in the domain [0, f_max]"""
        return int(self._f_max / self._delta_f) + 1

    @lru_cache()
    def __call__(self):
        """Array of uniform frequency bins in the domain [0, f_max]"""
        num_bins = self.__len__()
        sample_frequencies = np.linspace(0.0, self._f_max, num=num_bins, endpoint=True, dtype=np.float32)
        return sample_frequencies

    @property
    @lru_cache()
    def frequency_mask(self):
        """Mask which selects frequency bins greater than or equal to the starting frequency"""
        sample_frequencies = self.__call__()
        return sample_frequencies >= self._f_min

    @property
    def noise_std(self):
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        TODO: This description makes some assumptions that need to be clarified.
        """
        return np.sqrt(self._window_factor) / np.sqrt(4.0 * self._delta_f)

    def context_dim(self, num_detectors):
        # FIXME: "detectors" is astronomy specific; is there a better generic term?
        """Size of context data given the number of detectors."""
        num_bins_nonzero_data = int((self._f_max - self._f_min) / self._delta_f) + 1
        return num_bins_nonzero_data * num_detectors * 2

    # TODO: Do we want f_max to have a setter and getter and use sampling_rate internally as in the old code?
    @property
    def f_max(self):
        """The maximum frequency is set to half the sampling rate."""
        return self._f_max

    @f_max.setter
    def f_max(self, f_max):
        self._f_max = f_max

    @property
    def sampling_rate(self):
        """The sampling rate of the data."""
        return 2.0 * self._f_max

    @sampling_rate.setter
    def sampling_rate(self, fs):
        self._f_max = fs / 2.0


class TimeDomain(Domain):
    """Defines the physical time domain on which the data of interest live.

    The time bins are assumed to be uniform between [0, duration]
    with spacing 1 / sampling_rate.
    window_factor is used to compute noise_std().
    """

    def __init__(self, time_duration, sampling_rate):
        self._time_duration = time_duration
        self._sampling_rate = sampling_rate

    @lru_cache()
    def __len__(self):
        """Number of time bins given duration and sampling rate"""
        return int(self._time_duration * self._sampling_rate)

    @lru_cache()
    def __call__(self):
        """Array of uniform times at which data is sampled"""
        num_bins = self.__len__()
        return np.linspace(0.0, self._time_duration, num=num_bins,
                           endpoint=False, dtype=np.float32)

    @property
    def delta_t(self):
        """The size of the time bins"""
        return 1.0 / self._sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t):
        self._sampling_rate = 1.0 / delta_t

    @property
    def noise_std(self):
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        return 1.0 / np.sqrt(2.0 * self.delta_t)

    def context_dim(self, num_detectors):
        # FIXME: "detectors" is astronomy specific; is there a better generic term?
        """Size of context data given the number of detectors."""
        num_bins = self.__len__()
        return num_bins * num_detectors


class PCADomain(Domain):
    """TODO"""
    # FIXME: Should this be defined for FD or TD bases or both?

    # Nrb instead of Nf
    pass

    @property
    def noise_std(self):
        """Standard deviation of the whitened noise distribution.

        To have noise that comes from a multivariate *unit* normal
        distribution, you must divide by this factor. In practice, this means
        dividing the whitened waveforms by this.

        In the continuum limit in time domain, the standard deviation of white
        noise would at each point go to infinity, hence the delta_t factor.
        """
        # FIXME
        return np.sqrt(self.window_factor) / np.sqrt(4.0 * self.delta_f)

    def context_dim(self, num_detectors):
        # FIXME: "detectors" is astronomy specific; is there a better generic term?
        """Size of context data given the number of detectors."""
        num_rb = self.__len__()
        return num_rb * num_detectors * 2


class NonuniformFrequencyDomain(Domain):
    """TODO"""

    # It probably doesn't make sense to inherit from FrequencyDomain
    pass

