"""Settings for waveform generator configuration.

This module provides a serializable configuration class for waveform generation,
intended for use in YAML/JSON configuration files. For runtime parameters with
the domain object, use WaveformGeneratorParameters instead.
"""

from dataclasses import dataclass
from typing import Optional

from dingo.gw.approximant import Approximant
from dingo.gw.waveform_generator.waveform_generator_parameters import (
    _validate_waveform_generator_params,
)


@dataclass
class WaveformGeneratorSettings:
    """
    Serializable configuration settings for waveform generation.

    This class represents the configuration subset of WaveformGeneratorParameters
    that can be stored in YAML/JSON files. It excludes runtime-specific fields
    like the domain object, mode_list, lal_params, and transform.
    """

    approximant: Approximant
    f_ref: float
    spin_conversion_phase: Optional[float] = None
    f_start: Optional[float] = None

    def __post_init__(self):
        _validate_waveform_generator_params(
            self.approximant, self.f_ref, self.f_start
        )

    def to_dict(self):
        result = {
            "approximant": str(self.approximant),
            "f_ref": self.f_ref,
        }
        if self.spin_conversion_phase is not None:
            result["spin_conversion_phase"] = self.spin_conversion_phase
        if self.f_start is not None:
            result["f_start"] = self.f_start
        return result
