"""Time-domain strain posterior-predictive-distribution (PPD) plotting for GW results.

GW-specific counterpart to :mod:`dingo.core.utils.plotting`. At each time sample it takes
the highest-density interval (HDI) of the whitened strain ``h(t)`` over the
posterior-predictive draws produced by :meth:`dingo.gw.result.Result._compute_ppd`, and
fills between its edges. This is the *pointwise* credible band, as opposed to the central
percentile band shaded by bilby's ``plot_interferometer_waveform_posterior``.
"""

from typing import Dict, Optional, Tuple

import arviz as az
import numpy as np
from matplotlib import pyplot as plt

# One colour per posterior mode (its credible band and panel label); grey for the data.
_MODE_COLORS = ["#DD8452", "#4C72B0", "#55A868", "#C44E52", "#8172B3"]
_DATA_COLOR = "#555555"


def pointwise_hdi(
    td: np.ndarray,
    level: float,
    multimodal: bool = False,
) -> np.ndarray:
    """Pointwise highest-density interval (HDI) of ``p(h(t) | d)`` at each time sample.

    Thin wrapper over :func:`arviz.hdi`, which has no weights argument, so ``td`` must be
    equally weighted draws.

    Parameters
    ----------
    td : numpy.ndarray
        ``(n_draws, n_times)`` real whitened time-domain waveforms, equally weighted.
    level : float
        Credible level in ``(0, 1)``, e.g. ``0.9``.
    multimodal : bool
        Resolve a multimodal ``p(h(t)|d)`` into disjoint intervals instead of bridging the
        gap between the modes. Off by default: it estimates a KDE per time sample, which is
        both far slower and prone to splitting off spurious slivers on draws that are in
        practice unimodal at fixed ``t``.

    Returns
    -------
    numpy.ndarray
        ``(n_times, n_intervals, 2)`` interval edges, ascending in strain along axis 1 and
        NaN-padded where a time sample has fewer intervals than the widest one. The unimodal
        path always yields ``n_intervals == 1``.
    """
    td = np.asarray(td, dtype=float)
    # arviz wants (chain, draw, *shape): one chain, the PPD draws, then the time axis.
    intervals = np.asarray(az.hdi(td[None], hdi_prob=level, multimodal=multimodal))
    if not multimodal:
        intervals = intervals[:, None, :]
    return intervals


def _constant_mode_count_runs(intervals: np.ndarray):
    """Yield ``(slice, n_modes)`` for maximal time runs with a constant interval count.

    arviz sorts each column's intervals ascending, so slot ``k`` tracks the same branch
    throughout a run -- a run is the largest stretch fillable as a single polygon.
    """
    n_modes = np.isfinite(intervals[:, :, 0]).sum(axis=1)
    bounds = np.flatnonzero(np.diff(n_modes)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(n_modes)]))
    for start, stop in zip(starts, stops):
        yield slice(start, stop), int(n_modes[start])


def plot_ppd_td(
    wf_td: Dict[str, Dict[str, np.ndarray]],
    data_td: Dict[str, np.ndarray],
    times: np.ndarray,
    filename: str = "ppd_td.png",
    zoom: Optional[Tuple[float, float]] = None,
    strain_range: Optional[Tuple[float, float]] = None,
    hdi_level: float = 0.9,
    hdi_multimodal: bool = False,
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
    hdi_multimodal : bool
        Resolve disjoint HDI intervals per time sample. See :func:`pointwise_hdi`.

    Returns
    -------
    numpy.ndarray of the matplotlib Axes drawn onto (one per stacked (mode, detector) panel).
    """
    times = np.asarray(times, dtype=float)
    ifos = list(data_td.keys())
    modes = list(wf_td.keys())
    colors = {mode: _MODE_COLORS[i % len(_MODE_COLORS)] for i, mode in enumerate(modes)}
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
        color = colors[mode]
        band = np.asarray(wf_td[mode][ifo])[:, win]
        data = np.asarray(data_td[ifo])[win]

        lo, hi = (
            strain_range if strain_range is not None else _strain_range(band, data)
        )

        # Faint raw data underneath (grey noise), then the credible band on top.
        ax.plot(
            tt, data, color=_DATA_COLOR, lw=0.5, alpha=0.3, zorder=1, label="data",
        )
        intervals = pointwise_hdi(band, hdi_level, multimodal=hdi_multimodal)
        labelled = False
        # A separate fill per (run of constant mode count, interval slot), so a branch
        # splitting or merging breaks the fill instead of being bridged across.
        for sl, n_modes in _constant_mode_count_runs(intervals):
            for k in range(n_modes):
                lower, upper = intervals[sl, k, 0], intervals[sl, k, 1]
                lbl = f"{hdi_level:.0%} HDI" if (row == 0 and not labelled) else None
                labelled = True
                ax.fill_between(
                    tt[sl], lower, upper, color=color, alpha=0.30, lw=0, zorder=2,
                    label=lbl,
                )
                for edge in (lower, upper):
                    ax.plot(tt[sl], edge, color=color, lw=0.8, alpha=0.9, zorder=2)

        ax.set_xlim(*zoom)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("whitened strain")
        ax.text(
            0.01, 0.95, f"{mode} · {ifo}", transform=ax.transAxes,
            va="top", ha="left", fontweight="bold", color=color,
        )
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

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
