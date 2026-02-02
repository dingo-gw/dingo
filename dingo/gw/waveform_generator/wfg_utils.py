import numpy as np
import lal
import lalsimulation as LS
from scipy.signal import butter, sosfiltfilt
from lalsimulation.gwsignal.core import conditioning_subroutines as cond
import astropy.units as u
from gwpy.timeseries import TimeSeries


def linked_list_modes_to_dict_modes(hlm_ll):
    """Convert linked list of modes into dictionary with keys (l,m)."""
    hlm_dict = {}

    mode = hlm_ll.this
    while mode is not None:
        l, m = mode.l, mode.m
        hlm_dict[(l, m)] = mode.mode
        mode = mode.next

    return hlm_dict


def get_tapering_window_for_complex_time_series(h, tapering_flag: int = 1):
    """
    Get window for tapering of a complex time series from the lal backend. This is done
    by  tapering the time series with lal, and dividing tapered output by untapered
    input. lal does not support tapering of complex time series objects, so as a
    workaround we taper only the real part of the array and extract the window based on
    this.

    Parameters
    ----------
    h:
        complex lal time series object
    tapering_flag: int = 1
        Flag for tapering. See e.g. lines 2773-2777 in
            https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
            _l_a_l_sim_inspiral_waveform_taper_8c_source.html#l00222
        tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START

    Returns
    -------
    window: np.ndarray
        Array of length h.data.length, with the window used for tapering.
    """
    h_tapered = lal.CreateREAL8TimeSeries(
        "h_tapered", h.epoch, 0, h.deltaT, None, h.data.length
    )
    h_tapered.data.data = h.data.data.copy().real
    LS.SimInspiralREAL8WaveTaper(h_tapered.data, tapering_flag)
    # in case h.data.data is an array of 0's this makes the window return ones
    eps = 1e-20 * np.max(np.abs(h.data.data)) if np.max(np.abs(h.data.data)) > 0 else 1e-20
    window = (np.abs(h_tapered.data.data) + eps) / (np.abs(h.data.data.real) + eps)
    # FIXME: using eps for numerical stability is not really robust here
    return window


def taper_td_modes_in_place(hlm_td, tapering_flag: int = 1):
    """
    Taper the time domain modes in place.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding modes.
    tapering_flag: int = 1
        Flag for tapering. See e.g. lines 2773-2777 in
            https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
            _l_a_l_sim_inspiral_waveform_taper_8c_source.html#l00222
        tapering_flag = 1 corresponds to LAL_SIM_INSPIRAL_TAPER_START
    """
    for _, h in hlm_td.items():
        window = get_tapering_window_for_complex_time_series(h, tapering_flag)
        h.data.data *= window

def taper_td_modes_in_place_gwsignal(hlm_td, tapering_flag: str = 'start'):
    """
    Taper the time domain modes in place using gwsignal conditioning routines.
    There are apparently slight differences in the tapering functions 
    between these two and since some wfs (EG TEOB) uses the gwsignal 
    routine, we use the gwsignal routine sometimes. 

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding modes.

    """
    lalseries_to_gwpy_timeseries_in_place(hlm_td)
    for mode in hlm_td.keys():
        hlm_td[mode] = cond.taper_gwpy_timeseries(hlm_td[mode], tapering_flag)

def lalseries_to_gwpy_timeseries_in_place(hlm_td):
    """
    Convert lal time series in place to gwpy time series.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding modes.

    """
    from gwpy.timeseries import TimeSeries
    for mode in hlm_td.keys():
        times = (
            hlm_td[mode].epoch.gpsSeconds
            + hlm_td[mode].epoch.gpsNanoSeconds * 1e-9
            + np.arange(hlm_td[mode].data.length) * hlm_td[mode].deltaT
        )

        hlm_td[mode] = TimeSeries(
            data=hlm_td[mode].data.data,
            times=times,
            name=f"h_{mode[0]}_{mode[1]}",
            unit=u.dimensionless_unscaled,
        )

def gwpyseries_to_lalseries_in_place(hlm_td):
    """
    Convert gwpy time series in place to lal time series.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex gwpy time series objects for the
        corresponding modes.

    """
    for mode in hlm_td.keys():
        hlm_td[mode] = hlm_td[mode].to_lal()

def td_modes_to_fd_modes(hlm_td, domain):
    """
    Transform dict of td modes to dict of fd modes via FFT. The td modes are expected
    to be tapered.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and the complex lal time series objects for the
        corresponding tapered modes.
    domain: dingo.gw.domains.UniformFrequencyDomain
        Target domain after FFT.

    Returns
    -------
    hlm_fd: dict
        Dictionary with (l,m) keys and numpy arrays with the corresponding modes as
        values.
    """
    hlm_fd = {}

    delta_f = domain.delta_f
    delta_t = 0.5 / domain.f_max
    f_nyquist = domain.f_max  # use f_max as f_nyquist
    chirplen = int(2 * f_nyquist / delta_f)
    # sample frequencies, -f_max,...,-f_min,...0,...,f_min,...,f_max
    freqs = np.concatenate((-domain()[::-1], domain()[1:]), axis=0)
    # For even chirplength, we get chirplen + 1 output frequencies. However, the f_max
    # and -f_max bins are redundant, so we have chirplen unique bins.
    assert len(freqs) == chirplen + 1

    lal_fft_plan = lal.CreateForwardCOMPLEX16FFTPlan(chirplen, 0)
    for lm, h_td in hlm_td.items():
        assert np.abs(h_td.deltaT - delta_t) < 1e-12

        # resize data to chirplen by zero-padding or truncating
        # if chirplen < h_td.data.length:
        #     print(
        #         f"Specified frequency interval of {delta_f} Hz is too large "
        #         f"for a chirp of duration {h_td.data.length * delta_t} s with "
        #         f"Nyquist frequency {f_nyquist} Hz. The inspiral will be "
        #         f"truncated."
        #     )
        lal.ResizeCOMPLEX16TimeSeries(h_td, h_td.data.length - chirplen, chirplen)

        # Initialize a lal frequency series. We choose length chirplen + 1, while h_td is
        # only of length chirplen. This means, that the last bin h_fd.data.data[-1]
        # will not be modified by the lal FFT, and we have to copy over h_fd.data.data[0]
        # to h_fd.data.data[-1]. This corresponds to setting h(-f_max) = h(f_max).
        h_fd = lal.CreateCOMPLEX16FrequencySeries(
            "h_fd", h_td.epoch, 0, delta_f, None, chirplen + 1
        )
        # apply FFT
        lal.COMPLEX16TimeFreqFFT(h_fd, h_td, lal_fft_plan)
        assert np.abs(h_fd.deltaF - delta_f) < 1e-10
        assert np.abs(h_fd.f0 + domain.f_max) < 1e-6

        # time shift
        dt = (
            1.0 / h_fd.deltaF + h_fd.epoch.gpsSeconds + h_fd.epoch.gpsNanoSeconds * 1e-9
        )
        hlm_fd[lm] = h_fd.data.data * np.exp(-1j * 2 * np.pi * dt * freqs)
        # Set h(-f_max) = h(f_max), see above
        hlm_fd[lm][-1] = hlm_fd[lm][0]

    return hlm_fd

def td_modes_to_fd_modes_gwpy(hlm_td, domain):
    """
    Transform dict of gwpy TD modes to dict of one-sided FD modes via FFT.
    The td modes are expected to be conditioned (tapered/filtered).

    Uses numpy's complex FFT (not gwpy's rfft-based .fft()) to correctly
    handle complex-valued modes, then takes the positive-frequency half.
    Applies a time shift correction to account for the epoch (physical start
    time of the resized time series), matching the convention used by the
    LAL-based td_modes_to_fd_modes and the reference FD waveform path.

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and gwpy TimeSeries objects for the
        corresponding conditioned modes.
    domain: dingo.gw.domains.UniformFrequencyDomain
        Target domain after FFT.

    Returns
    -------
    hlm_fd: dict
        Dictionary with (l,m) keys and numpy arrays with the corresponding
        one-sided FD modes on [0, df, ..., f_max].
    """
    hlm_fd = {}

    delta_f = domain.delta_f
    delta_t = 0.5 / domain.f_max
    f_nyquist = domain.f_max
    chirplen = int(2 * f_nyquist / delta_f)
    frequency_array = domain()  # [0, df, ..., f_max]

    # Resize all modes and find consistent epoch from (2,2) mode
    resized_modes = {}
    epoch = None
    for (l, m), h in hlm_td.items():
        h_resized = cond.resize_gwpy_timeseries(h, len(h) - chirplen, chirplen)
        resized_modes[(l, m)] = h_resized
        # Use (2,2) mode (or first mode) to determine epoch
        if (l, m) == (2, 2) or epoch is None:
            peak_idx = np.argmax(np.abs(np.asarray(h_resized)))
            epoch = -peak_idx * delta_t

    # Precompute time shift (same for all modes)
    dt_shift = 1.0 / delta_f + epoch
    time_shift = np.exp(-1j * 2 * np.pi * dt_shift * frequency_array)

    for (l, m), h in resized_modes.items():
        data = np.asarray(h, dtype=complex)
        # Complex FFT -> take positive-frequency half
        fft_full = np.fft.fft(data, n=chirplen)
        fft_pos = fft_full[: chirplen // 2 + 1]
        # Standard continuous FT normalization: H(f) = dt * DFT
        hf = fft_pos * delta_t
        # Time shift correction to place merger at t=0
        hf *= time_shift
        hlm_fd[(l, m)] = hf

    return hlm_fd

def get_polarizations_from_fd_modes_m(hlm_fd, iota, phase):
    pol_m = {}
    polarizations = ["h_plus", "h_cross"]

    for (l, m), h in hlm_fd.items():
        if m not in pol_m:
            pol_m[m] = {k: 0.0 for k in polarizations}
            pol_m[-m] = {k: 0.0 for k in polarizations}

        # In the L0 frame, we compute the polarizations from the modes using the
        # spherical harmonics below.
        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m)
        ylmstar = ylm.conjugate()

        # Modes (l,m) are defined on domain -f_max,...,-f_min,...0,...,f_min,...,f_max.
        # This splits up the frequency series into positive and negative frequency parts.
        if len(h) % 2 != 1:
            raise ValueError(
                "Even number of bins encountered, should be odd: -f_max,...,0,...,f_max."
            )
        offset = len(h) // 2
        h1 = h[offset:]
        h2 = h[offset::-1].conj()

        # Organize the modes such that pol_m[m] transforms as e^{- 1j * m * phase}.
        # This differs from the usual way, e.g.,
        #   https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
        #   _l_a_l_sim_inspiral_8c_source.html#l04801
        pol_m[m]["h_plus"] += 0.5 * h1 * ylm
        pol_m[-m]["h_plus"] += 0.5 * h2 * ylmstar
        pol_m[m]["h_cross"] += 0.5 * 1j * h1 * ylm
        pol_m[-m]["h_cross"] += -0.5 * 1j * h2 * ylmstar

    return pol_m


def get_polarizations_from_onesided_fd_modes_m(hlm_fd, iota, phase):
    """Combine one-sided (f >= 0) FD modes into polarizations organized by m.

    For one-sided FD modes from non-precessing waveform models, the polarizations
    are recovered using the symmetry h_{l,-m}(t) = (-1)^l h*_{lm}(t), which
    relates the negative-frequency content to the positive-frequency modes:

        h+(f) = 0.5 sum_{l,m} [Y_{lm} + (-1)^l conj(Y_{l,-m})] H_{lm}(f)
        hx(f) = 0.5i sum_{l,m} [Y_{lm} - (-1)^l conj(Y_{l,-m})] H_{lm}(f)

    where Y_{lm} = _{-2}Y_{lm}(iota, pi/2 - phase).

    Each pol_m[m] transforms as exp(-1j * m * phase) under phase shifts.

    Parameters
    ----------
    hlm_fd : dict
        Dictionary with (l, m) keys and one-sided FD mode arrays (numpy arrays
        or gwpy FrequencySeries) as values, defined for f >= 0 only.
    iota : float
        Inclination angle.
    phase : float
        Reference phase.

    Returns
    -------
    pol_m : dict
        Dictionary with integer m as keys and
        {"h_plus": array, "h_cross": array} as values.
    """
    pol_m = {}
    polarizations = ["h_plus", "h_cross"]

    for (l, m), h in hlm_fd.items():
        if m not in pol_m:
            pol_m[m] = {k: 0.0 for k in polarizations}

        # Extract numpy array from gwpy FrequencySeries if needed
        h_data = h.value if hasattr(h, "value") else h

        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m)
        ylm_neg_conj = np.conj(
            lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, -m)
        )

        sign = (-1) ** l
        pol_m[m]["h_plus"] += 0.5 * (ylm + (sign * ylm_neg_conj)) * h_data
        pol_m[m]["h_cross"] += 0.5j * (ylm - (sign * ylm_neg_conj)) * h_data

    return pol_m


def get_starting_frequency_for_SEOBRNRv5_conditioning(parameters):
    """
    Compute starting frequency needed for having 3 extra cycles for tapering the TD modes.
    It returns the needed quantities to apply the standard LALSimulation conditioning routines to the TD modes.

    Parameters
    ----------
    parameters: dict
        Dictionary of parameters suited for GWSignal (obtained with NewInterfaceWaveformGenerator._convert_parameters)

    Returns
    ----------
    f_min: float
      Waveform starting frequency
    f_start: float
      New waveform starting frequency
    extra_time: float
      Extra time to take care of situations where the frequency is close to merger
    original_f_min: float
      Initial waveform starting frequency
    f_isco: float
      ISCO frequency
    """

    extra_time_fraction = (
        0.1  # fraction of waveform duration to add as extra time for tapering
    )
    extra_cycles = 3.0  # more extra time measured in cycles at the starting frequency

    f_min = parameters["f22_start"].value
    m1 = parameters["mass1"].value
    m2 = parameters["mass2"].value
    S1z = parameters["spin1z"].value
    S2z = parameters["spin2z"].value
    original_f_min = f_min

    f_isco = 1.0 / (pow(9.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)
    if f_min > f_isco:
        f_min = f_isco

    # upper bound on the chirp time starting at f_min
    tchirp = LS.SimInspiralChirpTimeBound(
        f_min, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, S1z, S2z
    )
    # upper bound on the final black hole spin */
    spinkerr = LS.SimInspiralFinalBlackHoleSpinBound(S1z, S2z)
    # upper bound on the final plunge, merger, and ringdown time */
    tmerge = LS.SimInspiralMergeTimeBound(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
    ) + LS.SimInspiralRingdownTimeBound((m1 + m2) * lal.MSUN_SI, spinkerr)

    # extra time to include for all waveforms to take care of situations where the frequency is close to merger (and is sweeping rapidly): this is a few cycles at the low frequency
    textra = extra_cycles / f_min
    # compute a new lower frequency
    f_start = LS.SimInspiralChirpStartFrequencyBound(
        (1.0 + extra_time_fraction) * tchirp + tmerge + textra,
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
    )

    f_isco = 1.0 / (pow(6.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)

    return f_min, f_start, extra_time_fraction * tchirp + textra, original_f_min, f_isco


def taper_td_modes_for_SEOBRNRv5_extra_time(
    h, extra_time, f_min, original_f_min, f_isco
):
    """
    Apply standard tapering procedure mimicking LALSimulation routine (https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_generator_conditioning_8c.html#ac78b5fcdabf8922a3ac479da20185c85)

    Parameters
    ----------
    h:
        complex gwpy TimeSeries object
    extra_time: float
        Extra time to take care of situations where the frequency is close to merger
    f_min: float
        Starting frequency employed in waveform generation
    original_f_min: float
        Initial starting frequency requested by the user
    f_isco:
        ISCO frequency

    Returns
    ----------
    h_return
        complex lal timeseries object
    """

    # Split in real and imaginary parts, since LAL conditioning routines are for real timeseries
    h_tapered_re = lal.CreateREAL8TimeSeries(
        "h_tapered", h.epoch.value, 0, h.dt.value, None, len(h)
    )
    h_tapered_re.data.data = h.value.copy().real

    h_tapered_im = lal.CreateREAL8TimeSeries(
        "h_tapered_im", h.epoch.value, 0, h.dt.value, None, len(h)
    )
    h_tapered_im.data.data = h.value.copy().imag

    # condition the time domain waveform by tapering in the extra time at the beginning and high-pass filtering above original f_min
    LS.SimInspiralTDConditionStage1(
        h_tapered_re, h_tapered_im, extra_time, original_f_min
    )
    # final tapering at the beginning and at the end to remove filter transients
    # waveform should terminate at a frequency >= Schwarzschild ISCO
    # so taper one cycle at this frequency at the end; should not make
    # any difference to IMR waveforms */
    LS.SimInspiralTDConditionStage2(h_tapered_re, h_tapered_im, f_min, f_isco)

    # Construct complex timeseries
    h_return = lal.CreateCOMPLEX16TimeSeries(
        "h_return",
        h_tapered_re.epoch,
        0,
        h_tapered_re.deltaT,
        None,
        h_tapered_re.data.length,
    )

    h_return.data.data = h_tapered_re.data.data + 1j * h_tapered_im.data.data

    # return timeseries
    return h_return


def get_conditioning_params_for_TEOB(parameters_gwsignal):
    """
    Compute conditioning parameters for TEOBResumSDALI waveforms.

    Replicates the parameter computation from gwsignal's
    generate_conditioned_td_waveform_from_td
    (lalsimulation/gwsignal/core/waveform_conditioning.py:34-111).

    This is needed because gwsignal's GenerateTDModes
    (lalsimulation/gwsignal/core/waveform.py:668) returns unconditioned modes,
    while GenerateFDWaveform routes through the conditioning pipeline. To make
    mode-decomposed waveforms match the reference, the same conditioning must be
    applied to each mode individually.

    Parameters
    ----------
    parameters_gwsignal: dict
        Dictionary of gwsignal parameters (from _convert_parameters).

    Returns
    -------
    f_start: float
        Lower starting frequency for generating modes with buffer for tapering.
    t_extra: float
        Duration of the extra time to taper at the beginning (for stage 1).
    f_min: float
        Effective minimum frequency, possibly lowered to fisco_9
        (for stage 2 beginning taper).
    original_f_min: float
        The originally requested starting frequency (for stage 1 high-pass).
    f_isco: float
        ISCO frequency at 6M (for stage 2 end taper).
    """
    # waveform_conditioning.py:51-52
    extra_time_fraction = 0.1
    extra_cycles = 3.0

    # waveform_conditioning.py:55-61
    # Masses are stored as astropy Quantities in solar masses; .value gives
    # solar masses, .si.value gives kg. LAL bound functions expect SI (kg).
    f_min = parameters_gwsignal["f22_start"].value
    m1_SI = parameters_gwsignal["mass1"].si.value
    m2_SI = parameters_gwsignal["mass2"].si.value
    s1z = parameters_gwsignal["spin1z"].value
    s2z = parameters_gwsignal["spin2z"].value
    original_f_min = f_min

    # waveform_conditioning.py:70-72 — clamp f_min to ISCO at 9^1.5
    fisco_9 = 1.0 / (
        np.power(9.0, 1.5) * np.pi * (m1_SI + m2_SI) * lal.MTSUN_SI / lal.MSUN_SI
    )
    if f_min > fisco_9:
        f_min = fisco_9

    # waveform_conditioning.py:76 — upper bound on chirp time from f_min
    tchirp = LS.SimInspiralChirpTimeBound(f_min, m1_SI, m2_SI, s1z, s2z)

    # waveform_conditioning.py:79
    s = LS.SimInspiralFinalBlackHoleSpinBound(s1z, s2z)

    # waveform_conditioning.py:82 — merger + ringdown time
    tmerge = (
        LS.SimInspiralMergeTimeBound(m1_SI, m2_SI)
        + LS.SimInspiralRingdownTimeBound(m1_SI + m2_SI, s)
    )

    # waveform_conditioning.py:87 — extra cycles near merger
    textra = extra_cycles / f_min

    # waveform_conditioning.py:90 — lower starting frequency for tapering buffer
    f_start = LS.SimInspiralChirpStartFrequencyBound(
        (1.0 + extra_time_fraction) * tchirp + tmerge + textra, m1_SI, m2_SI
    )

    # waveform_conditioning.py:106 — t_extra for stage 1 cosine taper
    t_extra = extra_time_fraction * tchirp + textra

    # waveform_conditioning.py:108 — ISCO at 6M for stage 2 end taper
    f_isco = 1.0 / (
        np.power(6.0, 1.5) * np.pi * (m1_SI + m2_SI) * lal.MTSUN_SI / lal.MSUN_SI
    )

    return f_start, t_extra, f_min, original_f_min, f_isco


def _high_pass_complex(data, dt, f_min, attenuation=0.99, order=8):
    """
    High-pass filter a complex time series using a Butterworth IIR filter.

    Replicates gwsignal's high_pass_time_series
    (lalsimulation/gwsignal/core/conditioning_subroutines.py:10-46).

    Parameters
    ----------
    data: np.ndarray (complex)
        Time series data array.
    dt: float
        Sampling interval in seconds.
    f_min: float
        Minimum frequency for high-pass.
    attenuation: float
        Attenuation at the low-frequency cutoff (default 0.99).
    order: int
        Order of Butterworth filter (default 8).

    Returns
    -------
    filtered: np.ndarray (complex)
        High-pass filtered data.
    """
    # conditioning_subroutines.py:33 — sampling frequency
    fs = 1.0 / dt

    # conditioning_subroutines.py:37-39 — bilinear transform to compute cutoff
    w1 = np.tan(np.pi * f_min * dt)
    wc = w1 * (1.0 / attenuation**0.5 - 1) ** (1.0 / (2.0 * order))
    fc = fs * np.arctan(wc) / np.pi

    # conditioning_subroutines.py:42-43 — forward-backward Butterworth filter
    sos = butter(order, fc, btype="highpass", output="sos", fs=fs)
    filtered_re = sosfiltfilt(sos, data.real)
    filtered_im = sosfiltfilt(sos, data.imag)
    return filtered_re + 1j * filtered_im


def _condition_stage1_complex(data, dt, t_extra, f_min):
    """
    Stage 1 conditioning: cosine taper at the beginning + high-pass filter.

    Replicates gwsignal's time_array_condition_stage1
    (lalsimulation/gwsignal/core/conditioning_subroutines.py:50-87).

    Parameters
    ----------
    data: np.ndarray (complex)
        Time series data array.
    dt: float
        Sampling interval in seconds.
    t_extra: float
        Duration of extra time at the beginning to taper.
    f_min: float
        Minimum frequency for high-pass filter.

    Returns
    -------
    data: np.ndarray (complex)
        Conditioned data.
    """
    # conditioning_subroutines.py:71-77 — cosine (Hann) taper at beginning
    Ntaper = int(np.round(t_extra / dt))
    if Ntaper > 0 and Ntaper < len(data):
        taper_array = np.arange(Ntaper)
        w = 0.5 - 0.5 * np.cos(taper_array * np.pi / Ntaper)
        data[:Ntaper] *= w

    # conditioning_subroutines.py:80-81 — high-pass filter
    data = _high_pass_complex(data, dt, f_min)

    # conditioning_subroutines.py:84-85 — trim trailing zeros
    data = np.trim_zeros(data, trim='b')

    return data


def _condition_stage2_complex(data, dt, f_min, f_isco):
    """
    Stage 2 conditioning: end taper (1 cycle at f_isco) + beginning taper
    (1 cycle at f_min).

    Replicates gwsignal's time_array_condition_stage2
    (lalsimulation/gwsignal/core/conditioning_subroutines.py:90-144).

    Parameters
    ----------
    data: np.ndarray (complex)
        Time series data array.
    dt: float
        Sampling interval in seconds.
    f_min: float
        Minimum frequency for beginning taper.
    f_isco: float
        ISCO frequency for end taper.

    Returns
    -------
    data: np.ndarray (complex)
        Conditioned data.
    """
    # conditioning_subroutines.py:111-114
    min_taper_samples = 4
    Nsize = len(data)
    if Nsize < 2 * min_taper_samples:
        return data

    # conditioning_subroutines.py:119-129 — end taper: 1 cycle at f_isco
    ntaper_end = max(int(np.round(1.0 / (f_isco * dt))), min_taper_samples)
    taper_array = np.arange(1, ntaper_end)
    w_end = 0.5 - 0.5 * np.cos(taper_array * np.pi / ntaper_end)
    data[Nsize - ntaper_end + 1 :] *= w_end[::-1]

    # conditioning_subroutines.py:133-142 — beginning taper: 1 cycle at f_min
    ntaper_begin = max(int(np.round(1.0 / (f_min * dt))), min_taper_samples)
    taper_array = np.arange(ntaper_begin)
    w_begin = 0.5 - 0.5 * np.cos(taper_array * np.pi / ntaper_begin)
    data[:ntaper_begin] *= w_begin

    return data


def condition_td_modes_for_TEOB_in_place(hlm_td, t_extra, f_min, original_f_min, f_isco):
    """
    Apply gwsignal-style conditioning to TD modes in place.

    This replicates the conditioning that gwsignal's
    generate_conditioned_td_waveform_from_td
    (lalsimulation/gwsignal/core/waveform_conditioning.py:34-111) applies to the
    combined TD polarizations, but applies it to individual complex modes. This
    is necessary because GenerateTDModes (gwsignal/core/waveform.py:668) returns
    unconditioned modes — it bypasses the conditioning pipeline entirely.

    The conditioning consists of two stages (following the C implementation at
    XLALSimInspiralTDConditionStage1 / Stage2):

    Stage 1 (conditioning_subroutines.py:50-87):
      - Cosine (Hann) taper over t_extra seconds at the beginning
      - Order-8 Butterworth high-pass filter at original_f_min (attenuation 0.99)

    Stage 2 (conditioning_subroutines.py:90-144):
      - End taper: 1 cycle at f_isco (ISCO at 6M)
      - Beginning taper: 1 cycle at f_min

    Usage
    -----
    The modes must be generated at the lower starting frequency f_start returned
    by get_conditioning_params_for_TEOB(), so that there is sufficient buffer
    for the tapering.

    Example::

        f_start, t_extra, f_min, original_f_min, f_isco = (
            get_conditioning_params_for_TEOB(parameters_gwsignal)
        )
        # Generate modes at f_start (not the original f_min)
        hlm_td, iota = generate_TD_modes_at_f_start(parameters, f_start)
        condition_td_modes_for_TEOB_in_place(
            hlm_td, t_extra, f_min, original_f_min, f_isco
        )
        hlm_fd = td_modes_to_fd_modes(hlm_td, domain)

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and complex LAL COMPLEX16TimeSeries objects.
    t_extra: float
        Duration of extra time at the beginning for stage 1 cosine taper.
        From get_conditioning_params_for_TEOB().
    f_min: float
        Effective minimum frequency (possibly lowered to fisco_9).
        Used for stage 2 beginning taper.
    original_f_min: float
        The originally requested starting frequency.
        Used for stage 1 high-pass filter.
    f_isco: float
        ISCO frequency at 6M. Used for stage 2 end taper.
    """
    for lm in list(hlm_td.keys()):
        h = hlm_td[lm]
        dt = h.dt.value
        data = h.value.copy()

        # waveform_conditioning.py:106 — stage 1 uses original_f_min for high-pass
        data = _condition_stage1_complex(data, dt, t_extra, original_f_min)

        # waveform_conditioning.py:109 — stage 2 uses adjusted f_min and f_isco
        data = _condition_stage2_complex(data, dt, f_min, f_isco)

        # Replace with new TimeSeries (length may have changed due to trim)
        hlm_td[lm] = TimeSeries(
            data, t0=h.t0, dt=h.dt, unit=h.unit
        )
