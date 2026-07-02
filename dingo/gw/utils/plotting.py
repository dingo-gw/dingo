"""Time-domain strain posterior-predictive-distribution (PPD) plotting for GW results.

GW-specific counterpart to :mod:`dingo.core.utils.plotting`: renders the whitened-strain
PPD envelopes produced by :meth:`dingo.gw.result.Result._compute_ppd` in the time domain.
:func:`plot_ppd_td` overlays one envelope per posterior mode in ``wf_fd`` (``"dingo"`` and,
when available, ``"dingo-is"``), mirroring how ``result.plot_corner`` shows both.

Inputs come straight from ``Result._compute_ppd``: ``wf_fd`` is
``{mode: {ifo: (n_waveforms, n_freq) complex}}`` (already whitened) and ``data_fd`` is
``{ifo: (n_freq,) complex}`` (whitened data); ``domain`` is the frequency ``Domain`` used
for the inverse FFT.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from dingo.gw.domains import Domain


def plot_ppd_td(
    wf_fd: dict,
    data_fd: dict,
    domain: Domain,
    filename: str = "ppd_td.png",
    zoom: Optional[Tuple[float, float]] = None,
    axes: Optional[Sequence[Axes]] = None,
    plot_data: bool = True,
    colors: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Plot the time-domain whitened-strain PPD, one envelope per posterior mode.

    For each detector, inverse-FFTs each mode's whitened waveforms to the time domain with
    the merger at t = 0 (segment-midpoint offset ``t0 = T/2``) and shades the pointwise
    min/max envelope. The grey whitened-data trace is overlaid once.

    Parameters
    ----------
    wf_fd : dict
        ``{mode: {ifo: (n_waveforms, n_freq) complex}}`` whitened model waveforms, from
        :meth:`Result._compute_ppd`. Each ``mode`` (e.g. ``"dingo"``, ``"dingo-is"``) is
        drawn as its own coloured, labelled envelope.
    data_fd : dict
        ``{ifo: (n_freq,) complex}`` whitened detector data; its keys set the subplot rows.
    domain : Domain
        Frequency domain used for the inverse FFT (needs ``delta_f``, ``f_max``, ``()``).
    filename : str
        Output path. Ignored when ``axes`` is supplied (the caller owns saving).
    zoom : tuple or None
        ``(left, right)`` x-limits in seconds-to-merger. Default ``(-1.0, 0.2)``.
    axes : sequence of matplotlib Axes or None
        Existing axes (one per detector) to draw onto for composition. When None a new
        figure is created and saved to ``filename``.
    plot_data : bool
        Draw the grey whitened-data trace once.
    colors : list[str] or None
        One color per mode; defaults to an internal cycle.

    Returns
    -------
    numpy.ndarray of the matplotlib Axes drawn onto.
    """
    # Envelope colors, one per overlaid posterior mode (first is the single-mode
    # default orange); grey for the whitened-data trace.
    ppd_colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    data_color = "#808080"

    ifos = list(data_fd.keys())
    modes = list(wf_fd.keys())
    if colors is None:
        colors = [ppd_colors[i % len(ppd_colors)] for i in range(len(modes))]

    if axes is None:
        fig, axes_col = plt.subplots(
            len(ifos), 1, figsize=(10, 3 * len(ifos)), sharex=True, squeeze=False
        )
        axes = axes_col[:, 0]
    else:
        fig = None
        axes = np.atleast_1d(axes)

    t0 = 1 / (2 * domain.delta_f)  # merger at segment midpoint -> t = 0
    phase_shift = np.exp(2j * np.pi * domain() * t0)

    for row, (ax, ifo) in enumerate(zip(axes, ifos)):
        for mode, color in zip(modes, colors):
            td = []
            for wf in wf_fd[mode][ifo] * phase_shift:
                times, x = one_sided_fd_to_td(wf, domain)
                td.append(np.real(x))
            td = np.array(td)
            ax.fill_between(
                times - t0,
                td.min(axis=0),
                td.max(axis=0),
                color=color,
                alpha=0.5,
                label=mode if row == 0 else None,
            )
        if plot_data:
            d_times, d_x = one_sided_fd_to_td(data_fd[ifo] * phase_shift, domain)
            d_td = np.convolve(np.real(d_x), np.ones(4) / 4, mode="same")
            ax.plot(
                d_times - t0,
                d_td,
                color=data_color,
                lw=1,
                alpha=0.7,
                zorder=0,
                label="data" if row == 0 else None,
            )
        ax.set_ylabel(f"{ifo}\nwhitened strain")
        ax.set_xlim(*(zoom if zoom is not None else (-1.0, 0.2)))

    axes[-1].set_xlabel("time to merger (s)")
    axes[0].legend(loc="upper left")
    if fig is not None:
        fig.tight_layout()
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
