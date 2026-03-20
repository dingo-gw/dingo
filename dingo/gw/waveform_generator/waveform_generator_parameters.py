from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import lal

from dingo.gw.approximant import Approximant
from dingo.gw.domains import Domain, build_domain
from dingo.gw.imports import read_file
from dingo.gw.logs import TableStr
from dingo.gw.types import Modes
from .polarizations import Polarization


def _validate_waveform_generator_params(
    approximant: Union[Approximant, str],
    f_ref: float,
    f_start: Optional[float] = None,
) -> Approximant:
    """Validate waveform generator parameters."""
    if f_ref <= 0:
        raise ValueError(f"f_ref must be positive, got {f_ref}")
    if f_start is not None and f_start <= 0:
        raise ValueError(f"f_start must be positive, got {f_start}")
    return approximant


@dataclass
class WaveformGeneratorParameters(TableStr):
    """
    Container class for parameters controlling gravitational waveform generation.

    This class is used at runtime and includes the Domain object.
    """

    approximant: Approximant
    domain: Domain
    f_ref: float
    f_start: Optional[float]
    spin_conversion_phase: Optional[float]
    mode_list: Optional[List[Modes]]
    lal_params: Optional[lal.Dict]
    transform: Optional[Callable[[Polarization], Polarization]] = None

    def __post_init__(self):
        _validate_waveform_generator_params(
            self.approximant, self.f_ref, self.f_start
        )

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], domain: Domain
    ) -> "WaveformGeneratorParameters":
        """Load parameters from file and combine with domain."""
        params = read_file(file_path)
        params["domain"] = domain
        return cls(**params)
