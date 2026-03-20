from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from dingo.gw.logs import TableStr


@dataclass
class WaveformParameters(TableStr):
    """
    Base class for waveform parameter sets.

    Subclass this for each approximant family. Use build_waveform_parameters()
    to instantiate the correct subclass based on the approximant name.
    """

    pass


@dataclass
class BBHWaveformParameters(WaveformParameters):
    """
    Parameters for binary black hole waveform generation.

    Used by all LALSimulation and GWSignal approximants. All parameters
    are optional and default to None.
    """

    luminosity_distance: Optional[float] = None
    redshift: Optional[float] = None
    comoving_distance: Optional[float] = None
    chi_1: Optional[float] = None
    chi_2: Optional[float] = None
    chi_1_in_plane: Optional[float] = None
    chi_2_in_plane: Optional[float] = None
    a_1: Optional[float] = None
    a_2: Optional[float] = None
    phi_jl: Optional[float] = None
    phi_12: Optional[float] = None
    tilt_1: Optional[float] = None
    tilt_2: Optional[float] = None
    dec: Optional[float] = None
    ra: Optional[float] = None

    geocent_time: Optional[float] = None

    delta_phase: Optional[float] = None
    phase: Optional[float] = None

    psi: Optional[float] = None
    theta_jn: Optional[float] = None

    # Mass parameters
    mass_1: Optional[float] = None
    mass_2: Optional[float] = None
    total_mass: Optional[float] = None
    chirp_mass: Optional[float] = None
    mass_ratio: Optional[float] = None
    symmetric_mass_ratio: Optional[float] = None

    # Source frame mass parameters
    mass_1_source: Optional[float] = None
    mass_2_source: Optional[float] = None
    total_mass_source: Optional[float] = None
    chirp_mass_source: Optional[float] = None

    l_max: Optional[float] = None

    # SEOBNRv5 specific parameters
    postadiabatic: Optional[Any] = None
    postadiabatic_type: Optional[Any] = None
    lmax_nyquist: Optional[int] = None


@dataclass
class RandomWaveformParameters(WaveformParameters):
    """
    Parameters for the RandomApproximant.

    Only requires mass_1, mass_2, luminosity_distance, and phase.
    All have sensible defaults.
    """

    mass_1: float = 30.0
    mass_2: float = 25.0
    luminosity_distance: float = 1000.0
    phase: float = 0.0


_APPROXIMANT_PARAMS_MAP: Dict[str, Type[WaveformParameters]] = {
    "RandomApproximant": RandomWaveformParameters,
}


def build_waveform_parameters(approximant: str, **kwargs: Any) -> WaveformParameters:
    """
    Factory: returns the correct parameter class for the given approximant.

    Parameters
    ----------
    approximant
        Approximant name (e.g. "IMRPhenomD", "RandomApproximant")
    **kwargs
        Parameter values to pass to the dataclass constructor

    Returns
    -------
    A WaveformParameters subclass instance appropriate for the approximant
    """
    cls = _APPROXIMANT_PARAMS_MAP.get(approximant, BBHWaveformParameters)
    return cls(**kwargs)
