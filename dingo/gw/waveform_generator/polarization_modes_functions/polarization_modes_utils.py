"""Utility functions for polarization modes.

Ported from dingo-waveform polarization_modes_functions/polarization_modes_utils.py.
"""

from typing import Dict

import lal
import lalsimulation as LS
import numpy as np

from dingo.gw.domains import BaseFrequencyDomain
from dingo.gw.types import FrequencySeries, Modes


def linked_list_modes_to_dict_modes(
    hlm_ll: LS.SphHarmFrequencySeries,
) -> Dict[Modes, lal.COMPLEX16FrequencySeries]:
    """Convert linked list of modes into dictionary with keys (l,m)."""
    hlm_dict = {}

    mode = hlm_ll.this
    while mode is not None:
        l, m = mode.l, mode.m
        hlm_dict[(l, m)] = mode.mode
        mode = mode.next

    return hlm_dict


def _get_tapering_window_for_complex_time_series(h, tapering_flag: int = 1):
    """
    Get window for tapering of a complex time series from the lal backend.

    This is done by tapering the time series with lal, and dividing tapered
    output by untapered input. lal does not support tapering of complex time
    series objects, so as a workaround we taper only the real part of the array
    and extract the window based on this.

    Parameters
    ----------
    h :
        complex lal time series object
    tapering_flag : int
        Flag for tapering. tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START

    Returns
    -------
    window : np.ndarray
        Array of length h.data.length, with the window used for tapering.
    """
    h_tapered = lal.CreateREAL8TimeSeries(
        "h_tapered", h.epoch, 0, h.deltaT, None, h.data.length
    )
    h_tapered.data.data = h.data.data.copy().real
    LS.SimInspiralREAL8WaveTaper(h_tapered.data, tapering_flag)
    eps = 1e-20 * np.max(np.abs(h.data.data))
    window = (np.abs(h_tapered.data.data) + eps) / (np.abs(h.data.data.real) + eps)
    return window


def taper_td_modes_in_place(
    hlm_td: Dict[Modes, lal.COMPLEX16FrequencySeries], tapering_flag: int = 1
) -> None:
    """
    Taper the time domain modes in place.

    Parameters
    ----------
    hlm_td : dict
        Dictionary with (l,m) keys and the complex lal time series objects.
    tapering_flag : int
        Flag for tapering. tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START
    """
    for _, h in hlm_td.items():
        window = _get_tapering_window_for_complex_time_series(h, tapering_flag)
        h.data.data *= window


def td_modes_to_fd_modes(
    hlm_td: Dict[Modes, lal.COMPLEX16FrequencySeries],
    domain: BaseFrequencyDomain,
) -> Dict[Modes, FrequencySeries]:
    """
    Transform dict of td modes to dict of fd modes via FFT.

    The td modes are expected to be tapered.

    Parameters
    ----------
    hlm_td : dict
        Dictionary with (l,m) keys and the complex lal time series objects.
    domain : BaseFrequencyDomain
        Target domain after FFT.

    Returns
    -------
    hlm_fd : dict
        Dictionary with (l,m) keys and numpy arrays with the corresponding modes.
    """
    hlm_fd: Dict[Modes, FrequencySeries] = {}

    domain_params = domain.get_parameters()

    if domain_params.delta_f is None:
        raise ValueError(
            "td_modes_to_fd_modes: frequency domain delta_f should not be None"
        )

    if domain_params.f_max is None:
        raise ValueError(
            "td_modes_to_fd_modes: frequency domain f_max should not be None"
        )

    delta_f = domain_params.delta_f
    delta_t = 0.5 / domain_params.f_max
    f_nyquist = domain_params.f_max
    n = round(f_nyquist / delta_f)
    if (n & (n - 1)) != 0:
        raise NotImplementedError("f_nyquist not a power of two of delta_f.")
    chirplen = int(2 * f_nyquist / delta_f)
    sample_frequencies = domain.sample_frequencies()
    freqs = np.concatenate((-sample_frequencies[::-1], sample_frequencies[1:]), axis=0)
    assert len(freqs) == chirplen + 1

    lal_fft_plan = lal.CreateForwardCOMPLEX16FFTPlan(chirplen, 0)
    for lm, h_td in hlm_td.items():
        delta_t_diff = np.abs(h_td.deltaT - delta_t)
        tolerance = 1e-3
        assert delta_t_diff < tolerance, (
            f"Mode {lm}: delta_t mismatch exceeds tolerance: "
            f"h_td.deltaT={h_td.deltaT}, expected={delta_t}, diff={delta_t_diff} > {tolerance}"
        )

        lal.ResizeCOMPLEX16TimeSeries(h_td, h_td.data.length - chirplen, chirplen)

        h_fd = lal.CreateCOMPLEX16FrequencySeries(
            "h_fd", h_td.epoch, 0, delta_f, None, chirplen + 1
        )
        lal.COMPLEX16TimeFreqFFT(h_fd, h_td, lal_fft_plan)
        delta_f_diff = np.abs(h_fd.deltaF - delta_f)
        delta_f_tolerance = max(1e-10, 0.02 * delta_f)
        assert delta_f_diff < delta_f_tolerance, (
            f"Mode {lm}: FFT output delta_f mismatch: "
            f"h_fd.deltaF={h_fd.deltaF}, expected={delta_f}, diff={delta_f_diff}"
        )
        f0_diff = np.abs(h_fd.f0 + domain_params.f_max)
        f0_tolerance = max(1e-6, 0.02 * domain_params.f_max)
        assert f0_diff < f0_tolerance, (
            f"Mode {lm}: FFT output f0 mismatch: "
            f"h_fd.f0={h_fd.f0}, expected={-domain_params.f_max}, diff={f0_diff}"
        )

        dt = (
            1.0 / h_fd.deltaF + h_fd.epoch.gpsSeconds + h_fd.epoch.gpsNanoSeconds * 1e-9
        )
        hlm_fd[lm] = h_fd.data.data * np.exp(-1j * 2 * np.pi * dt * freqs)
        hlm_fd[lm][-1] = hlm_fd[lm][0]

    return hlm_fd
