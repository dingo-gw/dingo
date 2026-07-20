"""Tests for the pointwise strain-PPD helpers in :mod:`dingo.gw.utils.plotting`."""

import numpy as np

from dingo.gw.utils.plotting import plot_ppd_td, pointwise_hdi


def test_pointwise_hdi_matches_normal_quantiles() -> None:
    """For a symmetric unimodal (normal) column the HDI is centred on the mode."""
    rng = np.random.default_rng(9)
    td = rng.normal(size=(20000, 3))
    intervals = pointwise_hdi(td, 0.5)
    assert intervals.shape == (3, 2)
    # Normal 50% HDI is [-0.674, 0.674] sigma, centred at 0.
    assert np.allclose(intervals[:, 0], -0.674, atol=0.06)
    assert np.allclose(intervals[:, 1], 0.674, atol=0.06)


def test_pointwise_hdi_is_narrowest_interval() -> None:
    """On a skewed column the HDI is narrower than the equal-tailed interval."""
    rng = np.random.default_rng(11)
    td = rng.exponential(size=(20000, 1))
    lower, upper = pointwise_hdi(td, 0.9)[0]
    quantiles = np.percentile(td[:, 0], [5.0, 95.0])
    assert upper - lower < quantiles[1] - quantiles[0]
    assert lower < 0.1  # the HDI of an exponential starts at the boundary


def test_pointwise_hdi_contains_the_requested_fraction() -> None:
    """The interval holds at least ``level`` of the draws at every time sample."""
    rng = np.random.default_rng(5)
    td = rng.normal(size=(1000, 25))
    lower, upper = pointwise_hdi(td, 0.9).T
    inside = ((td >= lower) & (td <= upper)).mean(axis=0)
    assert np.all(inside >= 0.9)


def test_pointwise_hdi_nested_levels() -> None:
    """The 50% interval is nested inside the 90% interval at every time sample."""
    rng = np.random.default_rng(3)
    td = rng.normal(size=(2000, 25))
    i90 = pointwise_hdi(td, 0.9)
    i50 = pointwise_hdi(td, 0.5)
    assert i90.shape == i50.shape == (25, 2)
    assert np.all(i90[:, 0] <= i50[:, 0] + 1e-9)
    assert np.all(i50[:, 1] <= i90[:, 1] + 1e-9)


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


def test_plot_ppd_td_labels_the_trigger_time(tmp_path) -> None:
    """The trigger GPS time is written into the title, at sub-sample precision."""
    rng = np.random.default_rng(8)
    T = 40
    times = np.linspace(-1.0, 0.2, T)
    args = ({"dingo": {"H1": rng.normal(size=(20, T))}}, {"H1": rng.normal(size=T)}, times)

    axes = plot_ppd_td(
        *args, filename=str(tmp_path / "titled.png"), trigger_time=1264316116.4
    )
    assert axes[0].figure._suptitle.get_text().endswith("GPS 1264316116.4000")
    assert axes[-1].get_xlabel() == "time relative to trigger (s)"

    axes = plot_ppd_td(*args, filename=str(tmp_path / "untitled.png"))
    assert "GPS" not in axes[0].figure._suptitle.get_text()


def test_plot_ppd_td_colors_are_per_mode(tmp_path) -> None:
    """Each mode gets its own band colour; dingo is blue and dingo-is orange by default."""
    rng = np.random.default_rng(7)
    n, T = 50, 40
    times = np.linspace(-1.0, 0.2, T)
    wf_td = {
        "dingo": {"H1": rng.normal(size=(n, T))},
        "dingo-is": {"H1": rng.normal(size=(n, T))},
    }
    axes = plot_ppd_td(
        wf_td, {"H1": rng.normal(size=T)}, times, filename=str(tmp_path / "colors.png")
    )
    dingo_fill, dingo_is_fill = (ax.collections[0].get_facecolor()[0] for ax in axes)
    assert dingo_fill[2] > dingo_fill[0]  # blue channel dominates
    assert dingo_is_fill[0] > dingo_is_fill[2]  # red channel dominates

    # The band edges are a darker shade of the band, and both are overridable.
    axes = plot_ppd_td(
        wf_td,
        {"H1": rng.normal(size=T)},
        times,
        filename=str(tmp_path / "override.png"),
        band_colors={"dingo": "#00FF00"},
        data_color="#FF0000",
    )
    data_line, *edge_lines = axes[0].lines
    assert data_line.get_color() == "#FF0000"
    assert axes[0].collections[0].get_facecolor()[0][1] == 1.0  # green band fill
    assert all(line.get_color() == (0.0, 0.65, 0.0) for line in edge_lines)


def test_plot_ppd_td_draws_overlay(tmp_path) -> None:
    """``plot_draws`` overlays the requested number of subsampled draw traces."""
    rng = np.random.default_rng(6)
    n, T = 200, 100
    times = np.linspace(-1.0, 0.2, T)
    wf_td = {"dingo": {"H1": rng.normal(size=(n, T))}}
    data_td = {"H1": rng.normal(size=T)}

    plain = plot_ppd_td(
        wf_td, data_td, times, filename=str(tmp_path / "plain.png")
    )
    with_draws = plot_ppd_td(
        wf_td,
        data_td,
        times,
        filename=str(tmp_path / "draws.png"),
        plot_draws=True,
        num_plotted_draws=20,
    )
    # 20 extra draw traces, and each is fainter than the band edges.
    assert len(with_draws[0].lines) == len(plain[0].lines) + 20
    assert max(line.get_alpha() for line in with_draws[0].lines[1:21]) < 0.3
