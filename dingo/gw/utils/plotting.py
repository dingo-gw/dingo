"""Time-domain strain posterior-predictive-distribution (PPD) plotting for GW results.

GW-specific counterpart to :mod:`dingo.core.utils.plotting`. At each time sample it takes
the highest-density interval (HDI) of the whitened strain ``h(t)`` over the
posterior-predictive draws produced by :meth:`dingo.gw.result.Result._compute_ppd`, and
fills between its edges. This is the *pointwise* credible band, as opposed to the central
percentile band shaded by bilby's ``plot_interferometer_waveform_posterior``.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt


def pointwise_hdi(td: np.ndarray, level: float) -> np.ndarray:
    """Pointwise highest-density interval (HDI) of ``p(h(t) | d)`` at each time sample.

    The narrowest interval containing a ``level`` fraction of the draws, found by sliding a
    window of that many draws over the sorted column and keeping the narrowest -- the
    unimodal algorithm of :func:`arviz.hdi`, vectorised over the time axis. It assumes
    ``p(h(t)|d)`` is unimodal at fixed ``t``, which whitened-strain draws are in practice;
    on a multimodal column it bridges the gap between the modes rather than resolving them.
    There is no weights argument, so ``td`` must be equally weighted draws.

    Parameters
    ----------
    td : numpy.ndarray
        ``(n_draws, n_times)`` real whitened time-domain waveforms, equally weighted.
    level : float
        Credible level in ``(0, 1)``, e.g. ``0.9``.

    Returns
    -------
    numpy.ndarray
        ``(n_times, 2)`` lower and upper interval edges.
    """
    td = np.sort(np.asarray(td, dtype=float), axis=0)
    n_draws = td.shape[0]
    # Number of sorted draws spanned by the interval; its two edges are that far apart.
    span = int(np.floor(level * n_draws))
    widths = td[span:] - td[: n_draws - span]
    start = np.argmin(widths, axis=0)
    cols = np.arange(td.shape[1])
    return np.stack((td[start, cols], td[start + span, cols]), axis=-1)


def plot_ppd_td(
    wf_td: Dict[str, Dict[str, np.ndarray]],
    data_td: Dict[str, np.ndarray],
    times: np.ndarray,
    filename: str = "ppd_td.png",
    hdi_level: float = 0.9,
    num_plotted_draws: int = 0,
    trigger_time: Optional[float] = None,
    band_colors: Optional[Dict[str, str]] = None,
    data_color: str = "#555555",
) -> np.ndarray:
    """Plot the time-domain whitened-strain PPD as pointwise credible bands.

    One panel per ``(mode, detector)``, stacked vertically. Each fills the pointwise HDI of
    ``p(h(t)|d)`` (:func:`pointwise_hdi`) over the raw whitened data, drawn as a faint grey
    trace on the whitened-noise scale (bilby/LVK convention). ``t = 0`` is the trigger.

    Everything is drawn over the full ``times`` array; the axes merely open on the last
    second of inspiral through ringdown, with the y-axis auto-scaled to the whitened noise
    over that view. Both limits are ordinary matplotlib state, so a caller wanting a
    different window sets it on the returned axes rather than through an argument here --
    the figure is closed once saved, but its axes stay live::

        axes = plot_ppd_td(wf_td, data_td, times)
        axes[0].set_xlim(-0.3, 0.1)  # sharex, so this moves every panel
        axes[0].figure.savefig("ppd_td_mergerzoom.png", dpi=200, bbox_inches="tight")

    Parameters
    ----------
    wf_td : dict
        ``{mode: {ifo: (n_draws, n_times) real}}`` whitened time-domain draws.
    data_td : dict
        ``{ifo: (n_times,) real}`` whitened detector data; its keys set the detectors.
    times : numpy.ndarray
        ``(n_times,)`` time axis in seconds relative to the trigger (``t = 0``).
    filename : str
        Output path for the saved figure.
    hdi_level : float
        Credible level of the filled band, in ``(0, 1)``.
    num_plotted_draws : int
        Overlay this many individual waveform draws, as faint traces underneath the band,
        taken as an evenly spaced subsample. Zero (the default) draws none: overlaying
        thousands is slow to render and mostly obscures the band.
    trigger_time : float or None
        GPS time that ``times`` is measured from, written into the title so the axis can be
        read back as an absolute time. ``None`` titles the figure generically.
    band_colors : dict or None
        ``{mode: colour}`` for the band, its draws and the panel label; merged over the
        defaults (blue for ``"dingo"``, orange for ``"dingo-is"``), with any further mode
        falling back to the matplotlib property cycle. The band's edges are drawn in a
        darker shade of it, so they stay legible where overlaid draws saturate the band.
    data_color : str
        Colour of the whitened detector data trace.

    Returns
    -------
    numpy.ndarray of the matplotlib Axes drawn onto (one per stacked (mode, detector) panel).
    """
    times = np.asarray(times, dtype=float)
    ifos = list(data_td.keys())
    modes = list(wf_td.keys())
    band_colors = {"dingo": "#4C72B0", "dingo-is": "#DD8452", **(band_colors or {})}
    colors = {
        mode: band_colors.get(mode, f"C{i}") for i, mode in enumerate(modes)
    }

    # The window the axes open on. times is monotone, so it is a contiguous slice -- index
    # with it rather than a boolean mask, to view the (n_draws, n_times) arrays instead of
    # copying them. Only the y-scaling reads it; the traces themselves are drawn in full.
    xlim = (-1.0, 0.2)
    start, stop = np.searchsorted(times, xlim)
    view = slice(start, stop) if stop > start else slice(None)

    panels = [(mode, ifo) for mode in modes for ifo in ifos]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for ax, (mode, ifo) in zip(axes, panels):
        band = np.asarray(wf_td[mode][ifo])
        data = np.asarray(data_td[ifo])
        band_color = colors[mode]
        edge_color = _darken(band_color)

        # Faint raw data underneath (grey noise), then the credible band on top.
        ax.plot(
            times, data, color=data_color, lw=0.5, alpha=0.3, zorder=1, label="data",
        )
        if num_plotted_draws > 0:
            step = max(1, len(band) // num_plotted_draws)
            for i, draw in enumerate(band[::step][:num_plotted_draws]):
                ax.plot(
                    times, draw, color=band_color, lw=0.4, alpha=0.15, zorder=2,
                    label="draws" if i == 0 else None,
                )
        lower, upper = pointwise_hdi(band, hdi_level).T
        ax.fill_between(
            times, lower, upper, color=band_color, alpha=0.30, lw=0, zorder=3,
            label=f"{hdi_level:.0%} HDI",
        )
        for edge in (lower, upper):
            ax.plot(times, edge, color=edge_color, lw=0.8, alpha=0.9, zorder=4)

        ax.set_xlim(*xlim)
        ax.set_ylim(*_strain_range(band[:, view], data[view]))
        ax.set_ylabel("whitened strain")
        ax.text(
            0.01, 0.95, f"{mode} · {ifo}", transform=ax.transAxes,
            va="top", ha="left", fontweight="bold", color=band_color,
        )
        # A legend per panel, not just the first: the modes differ in colour, so one
        # shared legend would show the wrong swatch for every panel below it.
        legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        # The traces are deliberately faint on the axes; opaque keys keep them readable.
        for handle in legend.get_lines():
            handle.set_alpha(1.0)

    axes[-1].set_xlabel("time relative to trigger (s)")
    fig.suptitle(
        "whitened strain PPD"
        if trigger_time is None
        else f"whitened strain PPD · trigger at GPS {trigger_time:.4f}"
    )
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return axes


def _darken(color: str, factor: float = 0.65) -> Tuple[float, float, float]:
    """Scale a colour's RGB channels towards black, for the band edges."""
    return tuple(factor * channel for channel in mcolors.to_rgb(color))


def _strain_range(band: np.ndarray, data: np.ndarray) -> Tuple[float, float]:
    """Robust, symmetric strain range on the whitened-noise scale.

    Set by the data's 99th ``|value|`` percentile (not its max, so a rare noise spike does
    not blow it up), widened if the reconstruction band would otherwise clip, padded by 10%.
    """
    hi = float(np.percentile(np.abs(np.asarray(data, dtype=float)), 99.0))
    env = np.percentile(np.abs(np.asarray(band, dtype=float)), 97.5, axis=0)
    if env.size:
        hi = max(hi, float(np.max(env)))
    hi = hi * 1.1 if hi > 0 else 1.0
    return -hi, hi
