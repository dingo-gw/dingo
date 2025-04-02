from abc import ABC, abstractmethod
from dingo.gw.gwutils import *


class Domain(ABC):
    """Defines the physical domain on which the data of interest live.

    This includes a specification of the bins or points,
    and a few additional properties associated with the data.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of bins or points in the domain"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Array of bins in the domain"""
        pass

    @abstractmethod
    def update(self, new_settings: dict):
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

    def __eq__(self, other):
        if self.domain_dict == other.domain_dict:
            return True
        else:
            return False
