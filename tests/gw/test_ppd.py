"""Tests for the pointwise strain-PPD helpers in :mod:`dingo.gw.utils.plotting`."""

import numpy as np

from dingo.gw.utils.plotting import (
    _constant_mode_count_runs,
    plot_ppd_td,
    pointwise_hdi,
)


def test_pointwise_hdi_unimodal_matches_normal_quantiles() -> None:
    """For a symmetric unimodal (normal) column the HDI is centred on the mode."""
    rng = np.random.default_rng(9)
    td = rng.normal(size=(20000, 3))
    intervals = pointwise_hdi(td, 0.5)
    assert intervals.shape == (3, 1, 2)  # unimodal -> one interval per time sample
    # Normal 50% HDI is [-0.674, 0.674] sigma, centred at 0.
    assert np.allclose(intervals[:, 0, 0], -0.674, atol=0.06)
    assert np.allclose(intervals[:, 0, 1], 0.674, atol=0.06)


def test_pointwise_hdi_splits_bimodal_column() -> None:
    """A bimodal column resolves into two disjoint intervals that skip the empty gap."""
    rng = np.random.default_rng(11)
    n = 4000
    bimodal = np.concatenate([rng.normal(-3, 0.3, n // 2), rng.normal(3, 0.3, n // 2)])
    td = bimodal[:, None]

    intervals = pointwise_hdi(td, 0.9, multimodal=True)
    assert intervals.shape == (1, 2, 2)  # two modes
    (lo1, hi1), (lo2, hi2) = intervals[0]
    assert lo1 < hi1 < lo2 < hi2  # disjoint and ascending
    assert hi1 < 0 < lo2  # the gap around 0 is excluded
    # The unimodal HDI, by contrast, bridges the gap.
    uni = pointwise_hdi(td, 0.9)  # unimodal is the default
    assert uni[0, 0, 0] < -3 < 0 < 3 < uni[0, 0, 1]


def test_pointwise_hdi_nested_levels() -> None:
    """The 50% interval is nested inside the 90% interval at every time sample."""
    rng = np.random.default_rng(3)
    td = rng.normal(size=(2000, 25))
    i90 = pointwise_hdi(td, 0.9)
    i50 = pointwise_hdi(td, 0.5)
    assert i90.shape == i50.shape == (25, 1, 2)  # unimodal draws
    assert np.all(i90[:, 0, 0] <= i50[:, 0, 0] + 1e-9)
    assert np.all(i50[:, 0, 1] <= i90[:, 0, 1] + 1e-9)


def test_constant_mode_count_runs_breaks_at_changes() -> None:
    """Runs split exactly where the number of resolved intervals changes."""
    nan = np.nan
    # mode counts over 6 time samples: 1, 1, 2, 2, 2, 1
    intervals = np.array(
        [
            [[0.0, 1.0], [nan, nan]],
            [[0.0, 1.0], [nan, nan]],
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 1.0], [nan, nan]],
        ]
    )
    runs = [(sl.start, sl.stop, n) for sl, n in _constant_mode_count_runs(intervals)]
    assert runs == [(0, 2, 1), (2, 5, 2), (5, 6, 1)]


def test_plot_ppd_td_renders_all_panels(tmp_path) -> None:
    """One panel per (mode, ifo); the figure writes and returns that many axes."""
    rng = np.random.default_rng(4)
    n, T = 200, 400
    times = np.linspace(-1.0, 0.2, T)
    wf_td = {
        "dingo": {"H1": rng.normal(size=(n, T)), "L1": rng.normal(size=(n, T))},
        "dingo-is": {"H1": rng.normal(size=(n, T)), "L1": rng.normal(size=(n, T))},
    }
    data_td = {"H1": rng.normal(size=T), "L1": rng.normal(size=T)}
    out = tmp_path / "ppd.png"
    axes = plot_ppd_td(wf_td, data_td, times, filename=str(out))
    assert out.exists()
    assert len(axes) == 4  # 2 modes x 2 detectors
