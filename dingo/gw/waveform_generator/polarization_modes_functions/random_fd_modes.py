"""
Random frequency-domain mode-separated polarization function for the RandomApproximant.

Generates synthetic mode-separated waveforms by decomposing the signal into
spherical harmonic contributions.
"""

from typing import Dict

import numpy as np

from dingo.gw.domains import BaseFrequencyDomain
from dingo.gw.waveform_generator.polarization_functions.random_fd import (
    _generate_waveform_array,
)
from dingo.gw.waveform_generator.polarizations import Polarization
from dingo.gw.types import Mode
from dingo.gw.waveform_generator.waveform_generator_parameters import (
    WaveformGeneratorParameters,
)
from dingo.gw.waveform_generator.waveform_parameters import RandomWaveformParameters

# Mode numbers and their relative amplitudes.
_MODE_CONFIG = {
    22: 1.0,  # dominant (2,2) mode
    33: 0.3,  # sub-dominant (3,3) mode
    44: 0.1,  # sub-dominant (4,4) mode
}


def random_inspiral_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: RandomWaveformParameters,
) -> Dict[Mode, Polarization]:
    """
    Generate synthetic mode-separated frequency-domain polarizations.

    Each mode is generated with a relative amplitude factor and a phase
    rotation of exp(-1j * m * phase).

    Parameters
    ----------
    waveform_gen_params
        Waveform generation configuration (domain, f_ref, etc.)
    waveform_params
        Waveform parameters (masses, distance, phase, etc.)

    Returns
    -------
    Dictionary mapping mode integer (e.g. 22, 33, 44) to Polarization
    """
    domain = waveform_gen_params.domain
    if not isinstance(domain, BaseFrequencyDomain):
        raise ValueError(
            f"random_inspiral_FD_modes requires a BaseFrequencyDomain, "
            f"got {type(domain).__name__}"
        )

    if waveform_params.phase is None:
        raise ValueError(
            "random_inspiral_FD_modes requires waveform_params.phase to be set"
        )

    sample_freqs_attr = getattr(domain, "sample_frequencies", None)
    frequencies = (
        sample_freqs_attr() if callable(sample_freqs_attr) else sample_freqs_attr
    )

    mass_1 = waveform_params.mass_1
    mass_2 = waveform_params.mass_2
    luminosity_distance = waveform_params.luminosity_distance
    phase = waveform_params.phase

    result: Dict[Mode, Polarization] = {}

    for mode, relative_amplitude in _MODE_CONFIG.items():
        m = mode % 10

        h_plus_base = _generate_waveform_array(
            frequencies, mass_1, mass_2, luminosity_distance, 0.0, domain.f_min
        )
        h_cross_base = _generate_waveform_array(
            frequencies, mass_1, mass_2, luminosity_distance, np.pi / 2.0, domain.f_min
        )

        h_plus_base *= relative_amplitude
        h_cross_base *= relative_amplitude

        phase_factor = np.exp(-1j * m * phase)
        h_plus_mode = h_plus_base * phase_factor
        h_cross_mode = h_cross_base * phase_factor

        result[mode] = Polarization(h_plus=h_plus_mode, h_cross=h_cross_mode)

    return result
