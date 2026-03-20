"""
Random frequency-domain polarization function for the RandomApproximant.

Generates synthetic waveforms with a physically motivated inspiral-like shape,
without calling any external waveform generation library.
"""

import numpy as np

from dingo.gw.domains import BaseFrequencyDomain
from dingo.gw.waveform_generator.polarizations import Polarization
from dingo.gw.waveform_generator.waveform_generator_parameters import (
    WaveformGeneratorParameters,
)
from dingo.gw.waveform_generator.waveform_parameters import RandomWaveformParameters

# Solar mass in kg, speed of light, gravitational constant (SI)
_MSUN_SI = 1.989e30
_C_SI = 2.998e8
_G_SI = 6.674e-11


def _compute_f_isco(mass_1: float, mass_2: float) -> float:
    """Compute the ISCO frequency for a binary with given component masses (solar masses)."""
    total_mass_kg = (mass_1 + mass_2) * _MSUN_SI
    return _C_SI**3 / (6**1.5 * np.pi * _G_SI * total_mass_kg)


def _generate_waveform_array(
    frequencies: np.ndarray,
    mass_1: float,
    mass_2: float,
    luminosity_distance: float,
    phase: float,
    f_min: float,
) -> np.ndarray:
    """
    Generate a single complex frequency-domain waveform array.

    The waveform has:
    - Amplitude: f^(-7/6) inspiral envelope, tapered near f_isco
    - Phase: smooth evolution seeded deterministically from the masses
    - Scaling: 1 / luminosity_distance
    """
    n = len(frequencies)
    h = np.zeros(n, dtype=np.complex128)

    f_isco = _compute_f_isco(mass_1, mass_2)

    mask = (frequencies >= f_min) & (frequencies > 0)
    if not np.any(mask):
        return h

    f_active = frequencies[mask]

    # Amplitude: inspiral power law with smooth cutoff at f_isco
    amplitude = f_active ** (-7.0 / 6.0)
    exponent = np.clip((f_active - f_isco) / (0.05 * f_isco), -50.0, 50.0)
    taper = 1.0 / (1.0 + np.exp(exponent))
    amplitude *= taper

    # Normalize so max amplitude is O(1e-21) at 1 Mpc
    amplitude *= 1e-21

    # Scale by distance
    if luminosity_distance > 0:
        amplitude /= luminosity_distance

    # Phase: deterministic smooth evolution seeded by the masses
    chirp_mass = (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2
    phase_evolution = -2.0 * np.pi * chirp_mass * (f_active / 100.0) ** (-5.0 / 3.0)
    phase_evolution += phase

    h[mask] = amplitude * np.exp(1j * phase_evolution)

    return h


def random_inspiral_FD(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: RandomWaveformParameters,
) -> Polarization:
    """
    Generate synthetic frequency-domain polarizations for RandomApproximant.

    Parameters
    ----------
    waveform_gen_params
        Waveform generation configuration (domain, f_ref, etc.)
    waveform_params
        Waveform parameters (masses, distance, phase, etc.)

    Returns
    -------
    Polarization with h_plus and h_cross arrays
    """
    domain = waveform_gen_params.domain
    if not isinstance(domain, BaseFrequencyDomain):
        raise ValueError(
            f"random_inspiral_FD requires a BaseFrequencyDomain, "
            f"got {type(domain).__name__}"
        )

    sample_freqs_attr = getattr(domain, "sample_frequencies", None)
    frequencies = sample_freqs_attr() if callable(sample_freqs_attr) else sample_freqs_attr

    mass_1 = waveform_params.mass_1
    mass_2 = waveform_params.mass_2
    luminosity_distance = waveform_params.luminosity_distance
    phase = waveform_params.phase

    h_plus = _generate_waveform_array(
        frequencies, mass_1, mass_2, luminosity_distance, phase, domain.f_min
    )

    h_cross = _generate_waveform_array(
        frequencies,
        mass_1,
        mass_2,
        luminosity_distance,
        phase + np.pi / 2.0,
        domain.f_min,
    )

    return Polarization(h_plus=h_plus, h_cross=h_cross)
