from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from dingo.gw.gwutils import *


@dataclass
class DomainParameters:
    """
    Dataclass representation of the domain.
    Domain instances can be created from instances of DomainParameters,
    or "saved" as instances of DomainParameters.

    The type, if not None, is expected to be an import path to a subclass
    of Domain. It will be used by the 'build_domain' function to instantiate
    the proper domain class.
    """

    f_max: Optional[float] = None
    delta_t: Optional[float] = None
    f_min: Optional[float] = None
    delta_f: Optional[float] = None
    window_factor: Optional[float] = None
    time_duration: Optional[float] = None
    sampling_rate: Optional[float] = None
    type: Optional[str] = None
    # MultibandedFrequencyDomain specific parameters
    nodes: Optional[list] = None
    delta_f_initial: Optional[float] = None
    base_delta_f: Optional[float] = None


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

    def get_parameters(self) -> DomainParameters:
        """
        Get the parameters of the domain as a DomainParameters dataclass.

        Returns
        -------
        DomainParameters
            The parameters of the domain.
        """
        raise NotImplementedError(
            "Subclasses of Domain must implement the get_parameters method."
        )

    @classmethod
    def from_parameters(cls, domain_parameters: DomainParameters) -> "Domain":
        """
        Create a domain instance from given parameters.

        Parameters
        ----------
        domain_parameters
            The parameters to create the domain.

        Returns
        -------
        Domain
            A corresponding instance of the domain.
        """
        raise NotImplementedError(
            "Subclasses of Domain must implement the from_parameters class method."
        )
