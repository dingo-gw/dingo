"""Time-domain strain posterior-predictive-distribution (PPD) plotting for GW results.

GW-specific counterpart to :mod:`dingo.core.utils.plotting`: renders the whitened-strain
PPD envelopes produced by :meth:`dingo.gw.result.Result._compute_ppd` in the time domain.
:func:`plot_ppd_td` draws one min/max envelope panel per (posterior mode, detector),
stacked vertically -- ``"dingo"`` panels on top and, when available, ``"dingo-is"`` below.

Inputs come straight from ``Result._compute_ppd``: ``wf_fd`` is
``{mode: {ifo: (n_waveforms, n_freq) complex}}`` (already whitened), ``data_fd`` is
``{ifo: (n_freq,) complex}`` (whitened data), and ``map_fd`` is
``{mode: {ifo: (n_freq,) complex}}`` (the per-mode maximum-probability waveform, drawn as
a line); ``domain`` is the frequency ``Domain`` used for the inverse FFT.
"""

from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from dingo.gw.domains import Domain

# One colour per posterior mode (first is the single-mode default orange); grey for data.
_PPD_COLORS = ["#DD8452", "#4C72B0", "#55A868", "#C44E52", "#8172B3"]
_DATA_COLOR = "#808080"


def plot_ppd_td(
    wf_fd: dict,
    data_fd: dict,
    domain: Domain,
    map_fd: Optional[dict] = None,
    level_counts: Optional[dict] = None,
    filename: str = "ppd_td.png",
    zoom: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Plot the time-domain whitened-strain PPD min/max envelope, one panel per (mode, ifo).

    For each mode and detector, inverse-FFTs the whitened draws to the time domain (merger at
    t = 0, segment-midpoint offset ``t0 = T/2``), shades the pointwise min/max envelope, and
    draws that mode's maximum-probability waveform as a line over it, all above the grey
    whitened-data trace. Panels are stacked vertically -- the **Dingo** posterior on top and,
    when the result is importance-sampled, **Dingo-IS** below. When ``level_counts`` gives more
    than one credible level, nested bands are drawn (tighter level = darker).

    Parameters
    ----------
    wf_fd : dict
        ``{mode: {ifo: (n_waveforms, n_freq) complex}}`` whitened model waveforms **ordered by
        descending density**, from :meth:`Result._compute_ppd`.
    data_fd : dict
        ``{ifo: (n_freq,) complex}`` whitened detector data; its keys set the detectors.
    domain : Domain
        Frequency domain used for the inverse FFT (needs ``delta_f``, ``f_max``, ``()``).
    map_fd : dict or None
        ``{mode: {ifo: (n_freq,) complex}}`` maximum-probability waveform per mode, drawn as
        a solid line in the mode's colour. Modes absent from the dict (or ``None``) get no line.
    level_counts : dict or None
        ``{mode: {level: k}}`` -- draw a min/max band over the densest ``k`` draws for each
        credible ``level``. ``None`` draws a single band over all draws.
    filename : str
        Output path for the saved figure.
    zoom : tuple or None
        ``(left, right)`` x-limits in seconds-to-merger. Default ``(-1.0, 0.2)``.

    Returns
    -------
    numpy.ndarray of the matplotlib Axes drawn onto (one per stacked (mode, detector) panel).
    """
    map_fd = map_fd or {}
    ifos = list(data_fd.keys())
    modes = list(wf_fd.keys())
    colors = {mode: _PPD_COLORS[i % len(_PPD_COLORS)] for i, mode in enumerate(modes)}
    zoom = zoom if zoom is not None else (-1.0, 0.2)

    t0 = 1 / (2 * domain.delta_f)  # merger at segment midpoint -> t = 0
    phase_shift = np.exp(2j * np.pi * domain() * t0)

    # Grey whitened-data trace per detector (shared across that detector's panels).
    data_td = {}
    for ifo in ifos:
        d_times, d_x = one_sided_fd_to_td(data_fd[ifo] * phase_shift, domain)
        data_td[ifo] = (d_times - t0, np.convolve(np.real(d_x), np.ones(4) / 4, mode="same"))

    # One panel per (mode, ifo), stacked vertically (grouped by mode: dingo, then dingo-is).
    panels = [(mode, ifo) for mode in modes for ifo in ifos]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for row, (ax, (mode, ifo)) in enumerate(zip(axes, panels)):
        color = colors[mode]
        td = np.array(
            [np.real(one_sided_fd_to_td(wf, domain)[1]) for wf in wf_fd[mode][ifo] * phase_shift]
        )
        times, _ = one_sided_fd_to_td(data_fd[ifo], domain)
        tt = times - t0

        # Levels largest -> smallest so the tighter (darker) band is drawn on top.
        counts = (level_counts or {}).get(mode, {}) or {None: len(td)}
        ordered = sorted(counts.items(), key=lambda kv: -(kv[0] or 1.0))
        multi = len(ordered) > 1

        # Grey data (bottom) -> nested min/max bands -> MAP line (top).
        d_t, d_td = data_td[ifo]
        ax.plot(d_t, d_td, color=_DATA_COLOR, lw=0.8, alpha=0.6, zorder=1)
        for i, (lvl, k) in enumerate(ordered):
            if k < 1:
                continue
            band = td[:k]
            ax.fill_between(
                tt, band.min(axis=0), band.max(axis=0),
                color=color, alpha=0.25 + 0.22 * i, zorder=2 + i,
                label=(f"{lvl:.0%} CI" if (multi and row == 0 and lvl is not None) else None),
            )
        if mode in map_fd:
            _, m_x = one_sided_fd_to_td(map_fd[mode][ifo] * phase_shift, domain)
            ax.plot(tt, np.real(m_x), color=color, lw=1.0, alpha=0.9, zorder=10)

        ax.set_xlim(*zoom)
        ax.set_ylabel("whitened strain")
        ax.text(
            0.01, 0.95, f"{mode} · {ifo}", transform=ax.transAxes,
            va="top", ha="left", fontweight="bold", color=color,
        )
        if multi and row == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("time to merger (s)")
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return axes


def one_sided_fd_to_td(
    fd: np.ndarray, domain: Domain
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-FFT a one-sided (positive-frequency) whitened spectrum to the time domain.

    Zeros the DC bin, then uses ``np.fft.irfft`` (which internally mirrors the conjugate
    spectrum) with a ``sqrt(N)`` normalization that matches Dingo's whitening convention
    (so whitened noise has unit variance). Returns the time array (``dt = 1 / (2 * f_max)``)
    together with the length ``2 * n - 1`` real time series.
    """
    fd = np.array(fd, dtype=np.complex128)
    fd[0] = 0.0  # zero DC

    n_time = 2 * fd.shape[0] - 1
    td = np.fft.irfft(fd, n=n_time) * np.sqrt(n_time)
    times = np.arange(n_time) * (1 / (2 * domain.f_max))

    return times, td
