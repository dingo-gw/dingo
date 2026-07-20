"""Time-domain strain posterior-predictive-distribution (PPD) plotting for GW results.

GW-specific counterpart to :mod:`dingo.core.utils.plotting`. At each time sample it takes
the highest-density interval (HDI) of the whitened strain ``h(t)`` over the
posterior-predictive draws produced by :meth:`dingo.gw.result.Result._compute_ppd`, and
fills between its edges. This is the *pointwise* credible band, as opposed to the central
percentile band shaded by bilby's ``plot_interferometer_waveform_posterior``.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

_BAND_COLOR = "#DD8452"
# Darker shade of the band colour for its edges, so they stay legible when the individual
# draws are overlaid and their faint traces saturate the band's interior.
_EDGE_COLOR = "#9C4C1C"
_DATA_COLOR = "#555555"


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
    zoom: Optional[Tuple[float, float]] = None,
    strain_range: Optional[Tuple[float, float]] = None,
    hdi_level: float = 0.9,
    plot_draws: bool = False,
    num_plotted_draws: int = 100,
) -> np.ndarray:
    """Plot the time-domain whitened-strain PPD as pointwise credible bands.

    One panel per ``(mode, detector)``, stacked vertically. Each fills the pointwise HDI of
    ``p(h(t)|d)`` (:func:`pointwise_hdi`) over the raw whitened data, drawn as a faint grey
    trace on the whitened-noise scale (bilby/LVK convention). ``t = 0`` is the network
    reference time.

    Parameters
    ----------
    wf_td : dict
        ``{mode: {ifo: (n_draws, n_times) real}}`` whitened time-domain draws.
    data_td : dict
        ``{ifo: (n_times,) real}`` whitened detector data; its keys set the detectors.
    times : numpy.ndarray
        ``(n_times,)`` time axis in seconds relative to the reference time (``t = 0``).
    filename : str
        Output path for the saved figure.
    zoom : tuple or None
        ``(left, right)`` x-limits in seconds relative to the reference time. Default
        ``(-1.0, 0.2)``.
    strain_range : tuple or None
        ``(low, high)`` y-limits (whitened strain). ``None`` auto-scales to the whitened
        noise; bound it tighter to zoom the y-axis onto the signal.
    hdi_level : float
        Credible level of the filled band, in ``(0, 1)``.
    plot_draws : bool
        Overlay the individual waveform draws as faint traces underneath the band. Off by
        default: with thousands of draws it is slow to render and mostly obscures the band.
    num_plotted_draws : int
        Number of draws overlaid when ``plot_draws``, taken as an evenly spaced subsample.

    Returns
    -------
    numpy.ndarray of the matplotlib Axes drawn onto (one per stacked (mode, detector) panel).
    """
    times = np.asarray(times, dtype=float)
    ifos = list(data_td.keys())
    modes = list(wf_td.keys())
    zoom = zoom if zoom is not None else (-1.0, 0.2)

    # times is monotone, so the zoom window is a contiguous slice -- index with it rather
    # than a boolean mask, to view the (n_draws, n_times) arrays instead of copying them.
    start, stop = np.searchsorted(times, zoom)
    win = slice(start, stop) if stop > start else slice(None)
    tt = times[win]

    panels = [(mode, ifo) for mode in modes for ifo in ifos]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for row, (ax, (mode, ifo)) in enumerate(zip(axes, panels)):
        band = np.asarray(wf_td[mode][ifo])[:, win]
        data = np.asarray(data_td[ifo])[win]

        lo, hi = (
            strain_range if strain_range is not None else _strain_range(band, data)
        )

        # Faint raw data underneath (grey noise), then the credible band on top.
        ax.plot(
            tt, data, color=_DATA_COLOR, lw=0.5, alpha=0.3, zorder=1, label="data",
        )
        if plot_draws:
            step = max(1, len(band) // num_plotted_draws)
            for i, draw in enumerate(band[::step][:num_plotted_draws]):
                ax.plot(
                    tt, draw, color=_BAND_COLOR, lw=0.4, alpha=0.08, zorder=2,
                    label="draws" if (row == 0 and i == 0) else None,
                )
        lower, upper = pointwise_hdi(band, hdi_level).T
        ax.fill_between(
            tt, lower, upper, color=_BAND_COLOR, alpha=0.30, lw=0, zorder=3,
            label=f"{hdi_level:.0%} HDI" if row == 0 else None,
        )
        for edge in (lower, upper):
            ax.plot(tt, edge, color=_EDGE_COLOR, lw=0.8, alpha=0.9, zorder=4)

        ax.set_xlim(*zoom)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("whitened strain")
        ax.text(
            0.01, 0.95, f"{mode} · {ifo}", transform=ax.transAxes,
            va="top", ha="left", fontweight="bold", color=_BAND_COLOR,
        )
        if row == 0:
            legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            # The traces are deliberately faint on the axes; opaque keys keep them readable.
            for handle in legend.get_lines():
                handle.set_alpha(1.0)

    axes[-1].set_xlabel("time relative to network reference time (s)")
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return axes


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
