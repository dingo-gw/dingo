"""
Plotting functions for mode-separated waveforms (from generate_hplus_hcross_m).

All functions return plotly Figure objects for interactive visualization.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dingo.gw.domains import Domain, BaseFrequencyDomain, TimeDomain
from dingo.gw.waveform_generator.polarizations import Polarization, sum_contributions_m
from dingo.gw.types import Mode


def plot_mode_amplitudes(
    modes: Dict[Mode, Polarization],
    domain: Domain,
    frequency: Optional[float] = None,
    polarization_type: Literal["plus", "cross"] = "plus",
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """
    Bar chart of mode amplitudes.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations from generate_hplus_hcross_m
    domain : Domain
        Domain specification
    frequency : float, optional
        Specific frequency to evaluate (Hz). If None, uses maximum over all frequencies.
    polarization_type : {'plus', 'cross'}
        Which polarization to analyze
    title : str, optional
        Plot title (auto-generated if None)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive bar chart with mode amplitudes
    """
    mode_labels = []
    amplitudes = []

    for m, polarization in sorted(modes.items()):
        mode_labels.append(f"m={m}")

        h = polarization.h_plus if polarization_type == "plus" else polarization.h_cross
        amp = np.abs(h)

        if frequency is not None and isinstance(domain, BaseFrequencyDomain):
            freqs = domain()
            idx = np.argmin(np.abs(freqs - frequency))
            amplitudes.append(amp[idx])
        else:
            amplitudes.append(np.max(amp))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=mode_labels,
            y=amplitudes,
            marker=dict(
                color=amplitudes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Amplitude"),
            ),
            hovertemplate="Mode: %{x}<br>Amplitude: %{y:.2e}<extra></extra>",
        )
    )

    if title is None:
        freq_str = f" at {frequency}Hz" if frequency is not None else " (max over frequency)"
        title = f"Mode Amplitudes ({polarization_type}){freq_str}"

    fig.update_layout(
        title=title,
        xaxis_title="Mode m",
        yaxis_title="Amplitude",
        yaxis_type="log",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_individual_modes(
    modes: Dict[Mode, Polarization],
    domain: Domain,
    mode_list: Optional[List[Mode]] = None,
    domain_type: Literal["time", "frequency"] = "frequency",
    polarization_type: Literal["plus", "cross"] = "plus",
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """
    Plot individual modes as separate traces on the same axes.

    Interactive legend allows toggling modes on/off.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations from generate_hplus_hcross_m
    domain : Domain
        Domain specification
    mode_list : List[Mode], optional
        Specific modes to plot. If None, plot all modes.
    domain_type : {'time', 'frequency'}
        Whether to plot in time or frequency domain
    polarization_type : {'plus', 'cross'}
        Which polarization to plot
    title : str, optional
        Plot title (auto-generated if None)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure with all modes
    """
    if mode_list is None:
        mode_list = sorted(modes.keys())

    fig = go.Figure()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    if domain_type == "frequency":
        x_values = domain()
        x_label = "Frequency (Hz)"

        for idx, m in enumerate(mode_list):
            polarization = modes[m]
            h = polarization.h_plus if polarization_type == "plus" else polarization.h_cross
            amp = np.abs(h)

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=amp,
                    mode="lines",
                    name=f"m={m}",
                    line=dict(color=colors[idx % len(colors)], width=1.5),
                    hovertemplate=f"m={m}<br>f=%{{x:.2f}}Hz<br>amp=%{{y:.2e}}<extra></extra>",
                )
            )

        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", title="Amplitude")

    else:  # time domain
        if not isinstance(domain, TimeDomain):
            raise ValueError("Time domain plotting requires TimeDomain object")

        x_values = domain()
        x_label = "Time (s)"

        for idx, m in enumerate(mode_list):
            polarization = modes[m]
            h = polarization.h_plus if polarization_type == "plus" else polarization.h_cross

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=np.real(h),
                    mode="lines",
                    name=f"m={m}",
                    line=dict(color=colors[idx % len(colors)], width=1.5),
                    hovertemplate=f"m={m}<br>t=%{{x:.3f}}s<br>h=%{{y:.2e}}<extra></extra>",
                )
            )

        fig.update_yaxes(title="Strain")

    if title is None:
        pol_name = "h<sub>+</sub>" if polarization_type == "plus" else "h<sub>x</sub>"
        title = f"Individual Modes ({pol_name}, {domain_type} domain)"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        height=height,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_mode_comparison(
    modes: Dict[Mode, Polarization],
    domain: Domain,
    reference_mode: Mode = (2, 2),
    metric: Literal["amplitude_ratio", "phase_difference"] = "amplitude_ratio",
    polarization_type: Literal["plus", "cross"] = "plus",
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """
    Compare all modes against a reference mode.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations
    domain : Domain
        Domain specification
    reference_mode : Mode
        Mode to use as reference for comparison
    metric : {'amplitude_ratio', 'phase_difference'}
        What to compare
    polarization_type : {'plus', 'cross'}
        Which polarization to analyze
    title : str, optional
        Plot title (auto-generated if None)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive comparison plot
    """
    if reference_mode not in modes:
        raise ValueError(f"Reference mode {reference_mode} not found in modes dictionary")

    ref_polarization = modes[reference_mode]
    ref_h = (
        ref_polarization.h_plus if polarization_type == "plus" else ref_polarization.h_cross
    )

    fig = go.Figure()

    x_values = domain()
    x_label = "Frequency (Hz)" if isinstance(domain, BaseFrequencyDomain) else "Time (s)"

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    idx = 0
    for m, polarization in sorted(modes.items()):
        if m == reference_mode:
            continue
        h = polarization.h_plus if polarization_type == "plus" else polarization.h_cross

        if metric == "amplitude_ratio":
            ratio = np.abs(h) / (np.abs(ref_h) + 1e-30)
            y_values = ratio
            y_label = f"Amplitude ratio (relative to {reference_mode})"
        else:  # phase_difference
            phase_diff = np.angle(h) - np.angle(ref_h)
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            y_values = phase_diff
            y_label = f"Phase difference (rad, relative to {reference_mode})"

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=f"m={m}",
                line=dict(color=colors[idx % len(colors)], width=1.5),
                hovertemplate=f"m={m}<br>{x_label.split()[0].lower()}=%{{x:.2f}}<br>value=%{{y:.2e}}<extra></extra>",
            )
        )
        idx += 1

    if title is None:
        pol_name = "h<sub>+</sub>" if polarization_type == "plus" else "h<sub>x</sub>"
        title = f"Mode Comparison: {metric.replace('_', ' ').title()} ({pol_name})"

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        hovermode="x unified",
        template="plotly_white",
    )

    if metric == "amplitude_ratio":
        fig.update_yaxes(type="log")

    if isinstance(domain, BaseFrequencyDomain):
        fig.update_xaxes(type="log")

    return fig


def plot_modes_grid(
    modes: Dict[Mode, Polarization],
    domain: Domain,
    domain_type: Literal["time", "frequency"] = "frequency",
    polarization_type: Literal["plus", "cross"] = "plus",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Grid of subplots, one per mode.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations
    domain : Domain
        Domain specification
    domain_type : {'time', 'frequency'}
        Whether to plot in time or frequency domain
    polarization_type : {'plus', 'cross'}
        Which polarization to plot
    title : str, optional
        Plot title (auto-generated if None)

    Returns
    -------
    plotly.graph_objects.Figure
        Grid plot with all modes
    """
    mode_list = sorted(modes.keys())
    n_modes = len(mode_list)

    n_cols = min(3, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols

    subplot_titles = [f"m={m}" for m in mode_list]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    if domain_type == "frequency":
        x_values = domain()
        x_label = "Frequency (Hz)"

        for idx, mode in enumerate(mode_list):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            polarization = modes[mode]
            h = polarization.h_plus if polarization_type == "plus" else polarization.h_cross
            amp = np.abs(h)

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=amp,
                    mode="lines",
                    name=f"m={mode}",
                    showlegend=False,
                    line=dict(color="#1f77b4", width=1.5),
                    hovertemplate="f=%{x:.2f}Hz<br>amp=%{y:.2e}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(type="log", row=row, col=col)
            fig.update_yaxes(type="log", row=row, col=col)

    else:  # time domain
        if not isinstance(domain, TimeDomain):
            raise ValueError("Time domain plotting requires TimeDomain object")

        x_values = domain()
        x_label = "Time (s)"

        for idx, mode in enumerate(mode_list):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            polarization = modes[mode]
            h = polarization.h_plus if polarization_type == "plus" else polarization.h_cross

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=np.real(h),
                    mode="lines",
                    name=f"m={mode}",
                    showlegend=False,
                    line=dict(color="#1f77b4", width=1.5),
                    hovertemplate="t=%{x:.3f}s<br>h=%{y:.2e}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    for col in range(1, n_cols + 1):
        fig.update_xaxes(title_text=x_label, row=n_rows, col=col)

    for row in range(1, n_rows + 1):
        y_label = "Amplitude" if domain_type == "frequency" else "Strain"
        fig.update_yaxes(title_text=y_label, row=row, col=1)

    if title is None:
        pol_name = "h<sub>+</sub>" if polarization_type == "plus" else "h<sub>x</sub>"
        title = f"Mode Grid ({pol_name}, {domain_type} domain)"

    fig.update_layout(
        title=title, height=300 * n_rows, width=400 * n_cols, template="plotly_white"
    )

    return fig


def plot_mode_reconstruction(
    modes: Dict[Mode, Polarization],
    domain: Domain,
    selected_modes: Optional[List[Mode]] = None,
    phase_shift: float = 0.0,
    domain_type: Literal["time", "frequency"] = "frequency",
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """
    Reconstruct total h+/hx from selected modes and compare to full sum.

    Parameters
    ----------
    modes : Dict[Mode, Polarization]
        Mode-separated polarizations
    domain : Domain
        Domain specification
    selected_modes : List[Mode], optional
        Modes to include in reconstruction. If None, use all modes.
    phase_shift : float
        Phase shift to apply when summing modes (radians)
    domain_type : {'time', 'frequency'}
        Whether to plot in time or frequency domain
    title : str, optional
        Plot title (auto-generated if None)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Comparison of partial and full reconstruction
    """
    pol_full = sum_contributions_m(modes, phase_shift=phase_shift)

    if selected_modes is not None:
        modes_subset = {m: modes[m] for m in selected_modes if m in modes}
        pol_partial = sum_contributions_m(modes_subset, phase_shift=phase_shift)
    else:
        pol_partial = pol_full
        selected_modes = list(modes.keys())

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("h<sub>+</sub>", "h<sub>x</sub>"),
        vertical_spacing=0.12,
    )

    x_values = domain()

    if domain_type == "frequency":
        x_label = "Frequency (Hz)"

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.abs(pol_full.h_plus),
                mode="lines",
                name="Full (all modes)",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="f=%{x:.2f}Hz<br>full=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.abs(pol_partial.h_plus),
                mode="lines",
                name=f"Partial ({len(selected_modes)} modes)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="f=%{x:.2f}Hz<br>partial=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.abs(pol_full.h_cross),
                mode="lines",
                name="Full (all modes)",
                showlegend=False,
                line=dict(color="#1f77b4", width=2),
                hovertemplate="f=%{x:.2f}Hz<br>full=%{y:.2e}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.abs(pol_partial.h_cross),
                mode="lines",
                name=f"Partial ({len(selected_modes)} modes)",
                showlegend=False,
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="f=%{x:.2f}Hz<br>partial=%{y:.2e}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(type="log", row=1, col=1)
        fig.update_xaxes(type="log", row=2, col=1)
        fig.update_yaxes(type="log", title="Amplitude", row=1, col=1)
        fig.update_yaxes(type="log", title="Amplitude", row=2, col=1)

    else:  # time domain
        if not isinstance(domain, TimeDomain):
            raise ValueError("Time domain plotting requires TimeDomain object")

        x_label = "Time (s)"

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.real(pol_full.h_plus),
                mode="lines",
                name="Full (all modes)",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="t=%{x:.3f}s<br>full=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.real(pol_partial.h_plus),
                mode="lines",
                name=f"Partial ({len(selected_modes)} modes)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="t=%{x:.3f}s<br>partial=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.real(pol_full.h_cross),
                mode="lines",
                name="Full (all modes)",
                showlegend=False,
                line=dict(color="#1f77b4", width=2),
                hovertemplate="t=%{x:.3f}s<br>full=%{y:.2e}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=np.real(pol_partial.h_cross),
                mode="lines",
                name=f"Partial ({len(selected_modes)} modes)",
                showlegend=False,
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="t=%{x:.3f}s<br>partial=%{y:.2e}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title="Strain", row=1, col=1)
        fig.update_yaxes(title="Strain", row=2, col=1)

    fig.update_xaxes(title_text=x_label, row=2, col=1)

    if title is None:
        title = f"Mode Reconstruction ({domain_type} domain)"

    fig.update_layout(title=title, height=height * 1.5, template="plotly_white")

    return fig
