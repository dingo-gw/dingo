from glob import escape
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
    domain: dingo.gw.domains.FrequencyDomain
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
    n = round(f_nyquist / delta_f)
    if (n & (n - 1)) != 0:
        raise NotImplementedError("f_nyquist not a power of two of delta_f.")
    chirplen = int(2 * f_nyquist / delta_f)
    # sample frequencies, -f_max,...,-f_min,...0,...,f_min,...,f_max
    freqs = np.concatenate((-domain()[::-1], domain()[1:]), axis=0)
    # For even chirplength, we get chirplen + 1 output frequencies. However, the f_max
    # and -f_max bins are redundant, so we have chirplen unique bins.
    assert len(freqs) == chirplen + 1

    lal_fft_plan = lal.CreateForwardCOMPLEX16FFTPlan(chirplen, 0)
    for lm, h_td in hlm_td.items():
        assert h_td.deltaT == delta_t

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
        assert h_fd.deltaF == delta_f
        assert h_fd.f0 == -domain.f_max

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

def get_polarizations_from_td_modes_m(hlm_td, iota, phase):
    # see https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_sph_harm_mode__h.html#ga834e376b58f9e71936ba14de58750a2b
    # Note this is not in the observer's frame but rather in the L0 frame

    pol_m = {}
    polarizations = ["h_plus", "h_cross"]

    for (l, m), hlm in hlm_td.items():
        if (l, m) not in pol_m:
            pol_m[(l, m)] = {k: 0.0 for k in polarizations}

            ylm = np.array(lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m))

            hpc = hlm.data.data*ylm
            h_plus = lal.CreateREAL8TimeSeries("h_plus", hlm.epoch, hlm.f0, hlm.deltaT, hlm.sampleUnits, hlm.data.length)
            h_plus.data.data = np.real(hpc)
            h_cross = lal.CreateREAL8TimeSeries("h_cross", hlm.epoch, hlm.f0, hlm.deltaT, hlm.sampleUnits, hlm.data.length)
            h_cross.data.data = -np.imag(hpc)

            pol_m[(l, m)]["h_plus"] = h_plus
            pol_m[(l, m)]["h_cross"] = h_cross
 
    return pol_m

def get_polarizations_from_td_modes_m_correct(hlm_td, iota, phase):
    # see https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_sph_harm_mode__h.html#ga834e376b58f9e71936ba14de58750a2b
    # Note this is not in the observer's frame but rather in the L0 frame

    pol_m = {}
    polarizations = ["h_plus", "h_cross"]

    for (l, m), hlm in hlm_td.items():
        if isinstance(hlm, np.ndarray):
            pass 
        else:
            hlm = hlm.data.data

        if (l, m) not in pol_m:
            pol_m[(l, m)] = {k: 0.0 for k in polarizations}
            pol_m[(l, -m)] = {k: 0.0 for k in polarizations}

            ylm = np.array(lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m))
            ylmstar = np.array(lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, -m))

            hpc = hlm*ylm
            pol_m[(l, m)]["h_plus"] += np.real(hpc)
            pol_m[(l, m)]["h_cross"] += -np.imag(hpc)
            
            if (l % 2):
                mult = -1
            else:
                mult = 1
            pol_m[(l, -m)]["h_plus"] += np.real(hpc)
            pol_m[(l, -m)]["h_cross"] += -np.imag(hpc)
 
    return pol_m

def get_aligned_spin_negative_modes_in_place(hlm_td):
    """
    Given a dict with (l, +m) as keys will return the dict with the (l, -m). This only works for aligned spins where
    h_lm = (-1)^l h*_{l -m}
    """
    for (l, m), hlm in hlm_td.copy().items():
        if (l % 2):
            f = -1
        else:
            f = 1
        
        name = hlm.name[:4] + "-" + hlm.name[4:]
        h_conj = lal.CreateCOMPLEX16TimeSeries(name, hlm.epoch, hlm.f0, hlm.deltaT, hlm.sampleUnits, hlm.data.length)
        h_conj.data.data = f*hlm.data.data.conj()
        hlm_td[(l, -m)] = h_conj

def correct_for_eob_lal_frame_rotation(h_plus, h_cross):
    """ Need to correct EOB to LAL wave frame due to different conventions
    https://git.ligo.org/waveforms/reviews/SEOBNRv4HM/-/blob/master/tests/conventions/conventions.pdf 
    """
    cp = np.cos(2*-np.pi / 2)
    sp = np.sin(2*-np.pi / 2)
    h_plus = cp*h_plus+ sp*h_cross
    h_cross = cp*h_cross - sp*h_plus
    return (h_plus, h_cross)

def taper_aligned_spin(h, m1, m2, extra_time_fraction, t_chirp, t_extra, f_min):
    # condition the time domain waveform by tapering in the extra time
    # at the beginning and high-pass filtering above original f_min
    
    if isinstance(h, dict) and "h_plus" in h.keys() and "h_cross" in h.keys():
        h_real, h_imag = h["h_plus"], h["h_cross"]
    else:
        # Technically not really h_plus and h_cross, they don't have the spherical harmonics
        h_real = lal.CreateREAL8TimeSeries("h_real", h.epoch, h.f0, h.deltaT, h.sampleUnits, h.data.length)
        h_real.data.data = np.real(h.data.data)
        h_imag = lal.CreateREAL8TimeSeries("h_imag", h.epoch, h.f0, h.deltaT, h.sampleUnits, h.data.length)
        h_imag.data.data = -np.imag(h.data.data)

    LS.SimInspiralTDConditionStage1(h_real, h_imag, extra_time_fraction * t_chirp + t_extra, f_min)

    # final tapering at the beginning and at the end to remove filter transients

    #  waveform should terminate at a frequency >= Schwarzschild ISCO
    #     * so taper one cycle at this frequency at the end; should not make
    #     * any difference to IMR waveforms
    fisco = 1.0 / ((6.0 ** 1.5) * np.pi * (m1 + m2) * (lal.MTSUN_SI / lal.MSUN_SI))
    LS.SimInspiralTDConditionStage2(h_real, h_imag, f_min, fisco)

    return h_real, h_imag

def get_stepped_back_f_start(f_min, m1, m2, S1z, S2z):
    extra_time_fraction = 0.1 # fraction of waveform duration to add as extra time for tapering
    extra_cycles = 3.0 # more extra time measured in cycles at the starting frequency

    # if the requested low frequency is below the lowest Kerr ISCO
    # frequency then change it to that frequency
    fisco = 1.0 / ((9.0 ** 1.5) * np.pi * (m1 + m2) * (lal.MTSUN_SI / lal.MSUN_SI))
    if f_min > fisco:
        f_min = fisco
    
    # upper bound on the chirp time starting at f_min
    t_chirp = LS.SimInspiralChirpTimeBound(f_min, m1, m2, S1z, S2z)

    # upper bound on the final black hole spin
    s = LS.SimInspiralFinalBlackHoleSpinBound(S1z, S2z)

    # upper bound on the final plunge, merger, and ringdown time 
    t_merge = LS.SimInspiralMergeTimeBound(m1, m2) + LS.SimInspiralRingdownTimeBound(m1 + m2, s)

    # extra time to include for all waveforms to take care of situations
    # where the frequency is close to merger (and is sweeping rapidly):
    # this is a few cycles at the low frequency
    t_extra = extra_cycles / f_min

    # time domain approximant: condition by generating a waveform
    # with a lower starting frequency and apply tapers in the
    # region between that lower frequency and the requested
    # frequency f_min; here compute a new lower frequency
    f_start = LS.SimInspiralChirpStartFrequencyBound((1.0 + extra_time_fraction) * t_chirp + t_merge + t_extra, m1, m2)
    
    return f_start, extra_time_fraction, t_chirp, t_extra

def taper_stepped_back_waveform_modes(hlm_td, m1, m2, extra_time_fraction, t_chirp, t_extra, f_min):
    for (l, m), hlm in hlm_td.copy().items():
        h_real, h_imag = taper_aligned_spin(hlm, m1, m2, extra_time_fraction, t_chirp, t_extra, f_min)
        strain = lal.CreateCOMPLEX16TimeSeries(f"h_{l,m}", hlm.epoch, hlm.f0, hlm.deltaT, hlm.sampleUnits, h_real.data.length)
        strain.data.data = h_real.data.data - 1j*h_imag.data.data
        hlm_td[(l, m)] = strain

    # Padding the end of modes with 0
    longest_arr_length = int(np.max([hlm.data.data.shape[0] for hlm in hlm_td.values()]))
    for (l, m), hlm in hlm_td.items():
        arr = np.pad(hlm.data.data, (0, longest_arr_length - hlm.data.data.shape[0]), 'constant', constant_values=(None, 0))
        hlm_td[(l, m)] = lal.CreateCOMPLEX16TimeSeries(f"h_{l,m}", hlm.epoch, hlm.f0, hlm.deltaT, hlm.sampleUnits, longest_arr_length)
        hlm_td[(l, m)].data.data = arr
    