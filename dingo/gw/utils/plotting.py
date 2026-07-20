"""Time-domain strain posterior-predictive-distribution (PPD) plotting for GW results.

GW-specific counterpart to :mod:`dingo.core.utils.plotting`: renders the *pointwise*
whitened-strain PPD produced by :meth:`dingo.gw.result.Result._compute_ppd` as filled
credible bands of ``p(h(t) | d)`` in the time domain.

Unlike bilby's :meth:`bilby.gw.result.CBCResult.plot_interferometer_waveform_posterior`,
which shades the *central* (percentile) credible band, and unlike the earlier Dingo
version, which shaded the min/max envelope of a highest-posterior-density set in
*parameter* space, this module works from the pointwise distribution: at each time sample
``t`` it takes the highest-density interval (HDI) of the whitened strain ``h(t)`` over the
posterior-predictive draws, via :func:`arviz.hdi`, and fills between its edges (90% by
default). ``arviz`` can also resolve a multimodal ``p(h(t)|d)`` into disjoint intervals,
which are then filled separately rather than bridged.

Inputs are time-domain arrays produced by ``Result._compute_ppd`` (which does the whitened
inverse FFT via the bilby :class:`~bilby.gw.detector.Interferometer`): ``wf_td`` is
``{mode: {ifo: (n_draws, n_times) real}}``, ``data_td`` is ``{ifo: (n_times,) real}``,
``weights`` is ``{mode: (n_draws,) or None}``, and ``times`` is the shared time axis in
seconds relative to the network reference time ``t_ref`` (so ``t = 0`` is ``t_ref``).
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

    Thin wrapper over :func:`arviz.hdi`. At each time sample this is the **shortest** set of
    strain intervals carrying ``level`` of the posterior mass -- the highest-density
    interval, not the central percentile band that bilby draws.

    ``multimodal`` resolves a multimodal ``p(h(t)|d)`` into *disjoint* intervals rather than
    bridging the low-density gap between the modes: for draws piled at +-3, the unimodal HDI
    is the single interval ``[-3.4, 3.4]`` -- which mostly covers the empty middle -- while
    the multimodal HDI is ``[-3.7, -2.3]`` and ``[2.4, 3.7]``. It is **off by default**: the
    unimodal path is exact (a shortest window over the sorted draws), whereas the multimodal
    path estimates a KDE per time sample, which is both far slower and fragile here. On real
    whitened-strain PPDs the pointwise distribution is unimodal at essentially every sample,
    and the multimodal path mostly splits off slivers ~5% the width of the main interval --
    spuriously, from KDE ripple. Duplicate draws make this much worse, since repeated rows
    spike the KDE; that is one reason :meth:`dingo.gw.result.Result._compute_ppd` builds its
    equally-weighted draws by *rejection* sampling rather than importance resampling. Turn it
    on only for a posterior genuinely expected to be multimodal at fixed ``t``, and check the
    intervals are comparable in width.

    ``arviz.hdi`` has no weights argument, so ``td`` must be **equally weighted** draws.
    ``Result._compute_ppd`` returns exactly that for every mode.

    Parameters
    ----------
    td : numpy.ndarray
        ``(n_draws, n_times)`` real whitened time-domain waveforms, equally weighted.
    level : float
        Credible level in ``(0, 1)``, e.g. ``0.9``.
    multimodal : bool
        Resolve disjoint intervals per time sample (see above). Off by default.

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
    """Yield ``(slice, n_modes)`` for maximal time runs over which ``n_modes`` is constant.

    Each run is one contiguous stretch where ``p(h(t)|d)`` resolves into the same number of
    disjoint HDI intervals. Since arviz sorts every column's intervals ascending, slot ``k``
    tracks the same branch throughout a run -- so a run is exactly the largest stretch that
    can be filled as one polygon, and mode-count changes (a branch splitting or merging) are
    where the fill must break rather than connect two different branches.
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
    weights: Optional[Dict[str, Optional[np.ndarray]]] = None,
    filename: str = "ppd_td.png",
    zoom: Optional[Tuple[float, float]] = None,
    strain_range: Optional[Tuple[float, float]] = None,
    hdi_level: float = 0.9,
    hdi_multimodal: bool = False,
    t_ref_label: str = "network reference time",
) -> np.ndarray:
    """Plot the time-domain whitened-strain PPD as pointwise credible bands.

    One panel per ``(mode, detector)``, stacked vertically -- the **dingo** posterior
    first and, when the result is importance-sampled, **dingo-is** below. Each panel fills
    the pointwise highest-density interval of ``p(h(t)|d)`` (:func:`pointwise_hdi`) as a
    shaded band with edge lines, over the raw whitened data drawn as a faint grey trace on
    its own (whitened-noise) scale -- the bilby/LVK convention, so the data reads as normal
    noise with the reconstruction band within it. Where ``p(h(t)|d)`` is multimodal (only
    with ``hdi_multimodal``) the band splits into disjoint filled intervals rather than
    bridging the low-density gap. ``t = 0`` is the network reference time.

    Parameters
    ----------
    wf_td : dict
        ``{mode: {ifo: (n_draws, n_times) real}}`` whitened time-domain draws.
    data_td : dict
        ``{ifo: (n_times,) real}`` whitened detector data; its keys set the detectors.
    times : numpy.ndarray
        ``(n_times,)`` time axis in seconds relative to the reference time (``t = 0``).
    weights : dict or None
        ``{mode: (n_draws,) or None}`` per-draw weights; a ``None`` value (or a missing
        mode) is uniform. Only used to auto-scale the strain axis --
        :meth:`dingo.gw.result.Result._compute_ppd` returns equally-weighted draws, since
        the HDI itself requires them.
    filename : str
        Output path for the saved figure.
    zoom : tuple or None
        ``(left, right)`` x-limits in seconds relative to the reference time (the time
        window to look at, e.g. around the merger). Default ``(-1.0, 0.2)``.
    strain_range : tuple or None
        ``(low, high)`` y-limits (whitened strain). ``None`` auto-scales to the whitened
        noise; bound it tighter to zoom the y-axis onto the signal.
    hdi_level : float
        Credible level of the filled band, in ``(0, 1)``.
    hdi_multimodal : bool
        Resolve disjoint HDI intervals per time sample (see :func:`pointwise_hdi`). Off by
        default: the whitened-strain PPD is unimodal at essentially every time sample, and
        arviz's multimodal path is KDE-based, so on these draws it mostly splits off
        negligible slivers rather than finding real modes -- and it is far slower. Turn it
        on for a posterior actually expected to be multimodal at fixed ``t``.
    t_ref_label : str
        Human-readable name of the ``t = 0`` reference, used in the axis label.

    Returns
    -------
    numpy.ndarray of the matplotlib Axes drawn onto (one per stacked (mode, detector) panel).
    """
    weights = weights or {}
    times = np.asarray(times, dtype=float)
    ifos = list(data_td.keys())
    modes = list(wf_td.keys())
    colors = {mode: _MODE_COLORS[i % len(_MODE_COLORS)] for i, mode in enumerate(modes)}
    zoom = zoom if zoom is not None else (-1.0, 0.2)

    win = (times >= zoom[0]) & (times <= zoom[1])
    if not win.any():  # zoom outside the data; fall back to the full span
        win = np.ones_like(times, dtype=bool)
    tt = times[win]

    panels = [(mode, ifo) for mode in modes for ifo in ifos]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for row, (ax, (mode, ifo)) in enumerate(zip(axes, panels)):
        color = colors[mode]
        w = weights.get(mode)
        band = np.asarray(wf_td[mode][ifo])[:, win]
        data = np.asarray(data_td[ifo])[win]

        # Strain (y) range. Default: the whitened-noise scale (bilby/LVK convention), so the
        # data reads as normal grey noise with the reconstruction band within it. An explicit
        # ``strain_range`` bounds it tighter, zooming the y-axis onto the signal.
        lo, hi = strain_range if strain_range is not None else _strain_range(
            band, data, weights=w
        )

        # Faint raw data underneath (grey noise), then the credible band on top.
        ax.plot(
            tt, data, color=_DATA_COLOR, lw=0.5, alpha=0.3, zorder=1, label="data",
        )
        intervals = pointwise_hdi(band, hdi_level, multimodal=hdi_multimodal)
        labelled = False
        # A separate fill per (run of constant mode count, interval slot): vector polygons,
        # so the band stays sharp at any zoom, and a branch splitting or merging breaks the
        # fill instead of being bridged across.
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

    axes[-1].set_xlabel(f"time relative to {t_ref_label} (s)")
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return axes


def _strain_range(
    band: np.ndarray, data: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Robust, symmetric strain range on the whitened-noise scale (bilby/LVK convention).

    The range is set by the whitened **data** (its 99th ``|value|`` percentile, not the raw
    max, so a rare noise spike does not blow it up) so the data reads as normal noise. It is
    widened only if the reconstruction band's near-merger bloom -- the max over time of the
    per-column 97.5th percentile of ``|h(t)|`` (IS-weighted for ``dingo-is``) -- would
    otherwise clip. Symmetrised about zero and padded by 10%.
    """
    hi = float(np.percentile(np.abs(np.asarray(data, dtype=float)), 99.0))
    band = np.abs(np.asarray(band, dtype=float))
    if weights is None:
        env = np.percentile(band, 97.5, axis=0)
    else:
        env = _weighted_quantile_columns(band, np.asarray(weights, float), 0.975)
    if env.size:
        hi = max(hi, float(np.max(env)))
    hi = hi * 1.1 if hi > 0 else 1.0
    return -hi, hi


def _weighted_quantile_columns(
    a: np.ndarray, weights: np.ndarray, q: float
) -> np.ndarray:
    """Per-column weighted ``q``-quantile of ``a`` (rows = draws, columns = time)."""
    order = np.argsort(a, axis=0)
    a_s = np.take_along_axis(a, order, axis=0)
    w_s = np.take_along_axis(np.broadcast_to(weights[:, None], a.shape), order, axis=0)
    cw = np.cumsum(w_s, axis=0)
    thresh = q * cw[-1, :]
    n_times = a.shape[1]
    out = np.empty(n_times)
    for t in range(n_times):
        out[t] = a_s[min(int(np.searchsorted(cw[:, t], thresh[t])), a.shape[0] - 1), t]
    return out
