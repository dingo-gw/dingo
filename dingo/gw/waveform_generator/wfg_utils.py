import numpy as np
import lal
import lalsimulation as LS
from scipy.signal import find_peaks
from lalsimulation.gwsignal.core import conditioning_subroutines as cond
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

def taper_td_gwpy_modes_in_place(hlm_td, iota, phase=0.0):
    """
    Apply sigmoid start taper to gwpy TimeSeries modes in place.

    This replicates the fallback conditioning used by gwsignal for
    approximants with f_ref_spin=False (e.g. TEOBResumSDALI).
    See generate_conditioned_td_waveform_from_td_fallback in
    lalsimulation/gwsignal/core/waveform_conditioning.py:35-64.

    The gwsignal reference applies taper_gwpy_timeseries to the combined
    polarizations hp and hc, where find_peaks determines the taper length.
    Since hp and hc can yield different taper lengths, but we need a single
    real-valued window to apply to each complex mode, we compute both windows
    and use the one with shorter taper (preserving more signal). 

    Parameters
    ----------
    hlm_td: dict
        Dictionary with (l,m) keys and gwpy TimeSeries objects.
    iota: float
        Inclination angle in radians.
    phase: float
        Reference phase for computing the combined polarizations used to
        determine the taper window. Defaults to 0.0 since modes should be
        phase-independent.
    """
    max_len = max(len(h) for h in hlm_td.values())

    # Sum modes to get combined hp and hc (unconditioned).
    # These are needed to compute taper windows that match the reference path,
    # where taper_gwpy_timeseries uses find_peaks on the combined polarizations.
    h_complex = np.zeros(max_len, dtype=complex)
    for (l, m), h in hlm_td.items():
        data = h.value.copy()
        if len(data) < max_len:
            data = np.pad(data, (0, max_len - len(data)))
        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m)
        h_complex += ylm * data

    hp_combined = h_complex.real
    hc_combined = -h_complex.imag

    # Compute separate taper windows for hp and hc. The reference path
    # (generate_conditioned_td_waveform_from_td_fallback) applies
    # taper_gwpy_timeseries independently to hp and hc, which can yield
    # different taper lengths because find_peaks finds different peaks.
    # We apply W_hp to Re(h_lm) and W_hc to Im(h_lm) for each mode.
    # For non-precessing systems the modes are purely real in the co-rotating
    # frame, so Re(h_lm) and Im(h_lm) map cleanly to the two polarizations,
    # giving exact agreement with the reference.
    start_hp, n_hp = _compute_sigmoid_taper_params(hp_combined)
    start_hc, n_hc = _compute_sigmoid_taper_params(hc_combined)

    if start_hp is None and start_hc is None:
        return

    # Build windows, falling back to the other if one polarization is empty
    if start_hp is None:
        start_hp, n_hp = start_hc, n_hc
    if start_hc is None:
        start_hc, n_hc = start_hp, n_hp

    W_hp = _compute_sigmoid_window(max_len, start_hp, n_hp)
    W_hc = _compute_sigmoid_window(max_len, start_hc, n_hc)

    for (l, m) in hlm_td:
        data = hlm_td[(l, m)].value.copy()
        if len(data) < max_len:
            data = np.pad(data, (0, max_len - len(data)))
        data = W_hp * data.real + 1j * W_hc * data.imag
        hlm_td[(l, m)] = TimeSeries(
            data,
            t0=hlm_td[(l, m)].t0,
            dt=hlm_td[(l, m)].dt,
        )

def _compute_sigmoid_taper_params(signal):
    """Compute (start, n) for the sigmoid start taper.

    Replicates the peak-finding logic in taper_gwpy_timeseries for
    taper_kind='start'. Returns (None, None) if the signal is empty
    or too short.
    """
    LALSIMULATION_RINGING_EXTENT = 19

    start = -1
    for idx, val in enumerate(signal):
        if val != 0:
            start = idx
            break
    if start == -1:
        return None, None

    end = -1
    for idx, val in enumerate(signal[::-1]):
        if val != 0:
            end = len(signal) - 1 - idx
            break

    if (end - start) <= 1:
        return None, None

    mid = int((start + end) / 2)
    pks, _ = find_peaks(abs(signal[start + 1 : mid]))
    pks = pks[pks > LALSIMULATION_RINGING_EXTENT]

    if len(pks) < 2:
        n = mid - start
    else:
        n = pks[1] + 1

    return start, n

def _compute_sigmoid_window(length, start, n):
    """Compute the sigmoid taper window matching taper_gwpy_timeseries."""
    window = np.ones(length)
    window[start] = 0.0
    realI = np.arange(1, n - 1)
    z = (n - 1.0) / realI + (n - 1.0) / (realI - (n - 1.0))
    sigma = 1.0 / (np.exp(z) + 1.0)
    window[start + 1 : start + n - 1] = sigma
    return window

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

    Replicates the FFT pipeline in gwsignal's
    generate_conditioned_fd_waveform_from_td (waveform_conditioning.py:510-522):
      resize → fft → normalize by 1/(2*df)

    No time shift is applied to the FD modes. The caller is responsible for
    computing and applying the time shift (which depends on the target phase
    via np.argmax of the reconstructed hp). This enables the "deferred time
    shift" approach where modes are FFT'd once (phase-independently) and the
    phase-dependent time shift is applied at resummation time.

    Also returns the resized TD mode data (numpy arrays) needed for the
    deferred time shift computation.

    Since the modes are complex-valued and gwpy's fft uses rfft (real input
    only), each mode is split into real and imaginary parts which are FFT'd
    separately and recombined: H_lm(f) = FFT(Re) + i*FFT(Im).

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
        one-sided FD modes on [0, df, ..., f_max]. No time shift applied.
    resized_td_modes: dict
        Dictionary with (l,m) keys and numpy complex arrays of the resized
        TD mode data, needed for deferred time shift computation.
    """
    hlm_fd = {}
    resized_td_modes = {}

    delta_f = domain.delta_f
    delta_t = 0.5 / domain.f_max
    f_nyquist = domain.f_max
    chirplen = int(2 * f_nyquist / delta_f)
    frequency_array = domain()  # [0, df, ..., f_max]

    for (l, m), h in hlm_td.items():
        # Resize to chirplen (waveform_conditioning.py:510)
        start_id = len(h) - chirplen
        h_resized = cond.resize_gwpy_timeseries(h, start_id, chirplen)
        resized_td_modes[(l, m)] = np.asarray(h_resized).copy()

        # Split complex mode into real and imaginary TimeSeries for gwpy fft
        h_re = TimeSeries(np.asarray(h_resized).real, dt=delta_t)
        h_im = TimeSeries(np.asarray(h_resized).imag, dt=delta_t)

        # FFT + normalize (waveform_conditioning.py:513-521)
        hf_re = h_re.fft()
        hf_re = hf_re / (2 * hf_re.df)

        hf_im = h_im.fft()
        hf_im = hf_im / (2 * hf_im.df)

        hlm_fd[(l, m)] = (hf_re.value[:len(frequency_array)]
                          + 1j * hf_im.value[:len(frequency_array)])

    return hlm_fd, resized_td_modes


def compute_epoch_from_resized_td_modes(resized_td_modes, iota, phase, delta_t):
    """Compute the epoch (time of peak) from resized TD modes at a given phase.

    Reconstructs the real h+ polarization from the resized TD mode data using
    spin-weighted spherical harmonics at the specified phase, then finds the
    peak via np.argmax. This matches the epoch convention used by gwsignal's
    resize_gwpy_timeseries (which calls np.argmax on the real hp).

    Parameters
    ----------
    resized_td_modes: dict
        Dictionary with (l,m) keys and numpy complex arrays of resized TD modes.
    iota: float
        Inclination angle in radians.
    phase: float
        Reference phase for spherical harmonic evaluation.
    delta_t: float
        Time step of the TD data.

    Returns
    -------
    epoch: float
        The epoch value (negative of peak index times delta_t).
    """
    chirplen = len(next(iter(resized_td_modes.values())))
    hp = np.zeros(chirplen)
    for (l, m), h_data in resized_td_modes.items():
        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m)
        hp += (ylm * h_data).real
    return -np.argmax(hp) * delta_t


def apply_time_shift_to_fd_modes(hlm_fd_raw, resized_td_modes, iota, phase, domain):
    """Apply a phase-dependent time shift to raw (unshifted) FD modes.

    Computes the epoch by reconstructing hp from resized TD modes at the given
    phase (matching gwsignal's resize_gwpy_timeseries convention), then applies
    the time shift exp(-i 2pi dt f) to each FD mode.

    Also returns a deferred_timeshift_data dict that can be passed to
    sum_contributions_m to correct the time shift when a phase_shift is applied.

    Parameters
    ----------
    hlm_fd_raw : dict
        Dictionary with (l,m) keys and numpy arrays of unshifted FD modes.
    resized_td_modes : dict
        Dictionary with (l,m) keys and numpy complex arrays of resized TD modes.
    iota : float
        Inclination angle in radians.
    phase : float
        Reference phase for epoch computation.
    domain : dingo.gw.domains.UniformFrequencyDomain
        Frequency domain.

    Returns
    -------
    hlm_fd : dict
        Dictionary with (l,m) keys and time-shifted FD mode arrays.
    deferred_timeshift_data : dict
        Data needed by sum_contributions_m for deferred time shift correction.
    """
    delta_t = 0.5 / domain.f_max
    delta_f = domain.delta_f
    frequency_array = domain()

    epoch = compute_epoch_from_resized_td_modes(
        resized_td_modes, iota, phase, delta_t
    )
    dt = 1.0 / delta_f + epoch
    time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)

    hlm_fd = {lm: hf * time_shift for lm, hf in hlm_fd_raw.items()}

    deferred_timeshift_data = {
        "resized_td_modes": resized_td_modes,
        "iota": iota,
        "phase_ref": phase,
        "dt_ref": dt,
        "delta_t": delta_t,
        "delta_f": delta_f,
        "frequency_array": frequency_array,
    }

    return hlm_fd, deferred_timeshift_data


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


