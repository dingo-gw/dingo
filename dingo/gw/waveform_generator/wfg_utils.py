import numpy as np
import lal
import lalsimulation as LS


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
    eps = 1e-20 * np.max(np.abs(h.data.data))
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
