"""
Plotting functions for Polarization objects (from generate_hplus_hcross).

All functions return plotly Figure objects for interactive visualization.
"""

from typing import Literal, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dingo.gw.domains import Domain, TimeDomain, BaseFrequencyDomain
from dingo.gw.waveform_generator.polarizations import Polarization
from .converters import polarization_to_gwpy_timeseries, polarization_to_gwpy_frequencyseries


def plot_polarizations_time(
    polarization: Polarization,
    domain: TimeDomain,
    title: str = "Gravitational Wave Polarizations (Time Domain)",
    show_both: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Interactive time-domain plot of h+ and hx strain.

    Parameters
    ----------
    polarization : Polarization
        Polarization object containing h_plus and h_cross
    domain : TimeDomain
        Time domain specification
    title : str
        Plot title
    show_both : bool
        If True, show both h+ and hx on same plot. If False, create subplots.
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    times = domain()

    if show_both:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.real(polarization.h_plus),
                mode="lines",
                name="h<sub>+</sub>",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="t=%{x:.3f}s<br>h<sub>+</sub>=%{y:.2e}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.real(polarization.h_cross),
                mode="lines",
                name="h<sub>x</sub>",
                line=dict(color="#ff7f0e", width=1.5),
                hovertemplate="t=%{x:.3f}s<br>h<sub>x</sub>=%{y:.2e}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Strain",
            height=height,
            hovermode="x unified",
            template="plotly_white",
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("h<sub>+</sub>", "h<sub>x</sub>"),
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.real(polarization.h_plus),
                mode="lines",
                name="h<sub>+</sub>",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="t=%{x:.3f}s<br>h<sub>+</sub>=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.real(polarization.h_cross),
                mode="lines",
                name="h<sub>x</sub>",
                line=dict(color="#ff7f0e", width=1.5),
                hovertemplate="t=%{x:.3f}s<br>h<sub>x</sub>=%{y:.2e}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Strain", row=1, col=1)
        fig.update_yaxes(title_text="Strain", row=2, col=1)

        fig.update_layout(
            title=title, height=height * 1.5, template="plotly_white", showlegend=False
        )

    return fig


def plot_polarizations_frequency(
    polarization: Polarization,
    domain: BaseFrequencyDomain,
    title: str = "Gravitational Wave Polarizations (Frequency Domain)",
    plot_type: Literal["amplitude", "phase", "both"] = "amplitude",
    log_scale: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Interactive frequency-domain plot with amplitude and/or phase.

    Parameters
    ----------
    polarization : Polarization
        Polarization object containing frequency-domain h_plus and h_cross
    domain : BaseFrequencyDomain
        Frequency domain specification
    title : str
        Plot title
    plot_type : {'amplitude', 'phase', 'both'}
        What to plot: amplitude, phase, or both
    log_scale : bool
        Use log scale for y-axis (amplitude) and x-axis (frequency)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    frequencies = domain()

    amp_plus = np.abs(polarization.h_plus)
    amp_cross = np.abs(polarization.h_cross)
    phase_plus = np.angle(polarization.h_plus)
    phase_cross = np.angle(polarization.h_cross)

    if plot_type == "amplitude":
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=amp_plus,
                mode="lines",
                name="|h<sub>+</sub>|",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="f=%{x:.2f}Hz<br>|h<sub>+</sub>|=%{y:.2e}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=amp_cross,
                mode="lines",
                name="|h<sub>x</sub>|",
                line=dict(color="#ff7f0e", width=1.5),
                hovertemplate="f=%{x:.2f}Hz<br>|h<sub>x</sub>|=%{y:.2e}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude",
            height=height,
            hovermode="x unified",
            template="plotly_white",
        )

        if log_scale:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")

    elif plot_type == "phase":
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=phase_plus,
                mode="lines",
                name="ang h<sub>+</sub>",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="f=%{x:.2f}Hz<br>ang h<sub>+</sub>=%{y:.2f}rad<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=phase_cross,
                mode="lines",
                name="ang h<sub>x</sub>",
                line=dict(color="#ff7f0e", width=1.5),
                hovertemplate="f=%{x:.2f}Hz<br>ang h<sub>x</sub>=%{y:.2f}rad<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Phase (rad)",
            height=height,
            hovermode="x unified",
            template="plotly_white",
        )

        if log_scale:
            fig.update_xaxes(type="log")

    else:  # both
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Amplitude", "Phase"),
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=amp_plus,
                mode="lines",
                name="|h<sub>+</sub>|",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="f=%{x:.2f}Hz<br>|h<sub>+</sub>|=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=amp_cross,
                mode="lines",
                name="|h<sub>x</sub>|",
                line=dict(color="#ff7f0e", width=1.5),
                hovertemplate="f=%{x:.2f}Hz<br>|h<sub>x</sub>|=%{y:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=phase_plus,
                mode="lines",
                name="ang h<sub>+</sub>",
                line=dict(color="#1f77b4", width=1.5),
                showlegend=False,
                hovertemplate="f=%{x:.2f}Hz<br>ang h<sub>+</sub>=%{y:.2f}rad<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=phase_cross,
                mode="lines",
                name="ang h<sub>x</sub>",
                line=dict(color="#ff7f0e", width=1.5),
                showlegend=False,
                hovertemplate="f=%{x:.2f}Hz<br>ang h<sub>x</sub>=%{y:.2f}rad<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)

        if log_scale:
            fig.update_xaxes(type="log", row=1, col=1)
            fig.update_xaxes(type="log", row=2, col=1)
            fig.update_yaxes(type="log", row=1, col=1)

        fig.update_layout(title=title, height=height * 1.5, template="plotly_white")

    return fig


def plot_polarization_spectrogram(
    polarization: Polarization,
    domain: Domain,
    polarization_type: Literal["plus", "cross", "both"] = "plus",
    fftlength: float = 0.1,
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """
    Time-frequency spectrogram using Short-Time Fourier Transform.

    Parameters
    ----------
    polarization : Polarization
        Polarization object
    domain : Domain
        Domain specification (must be TimeDomain)
    polarization_type : {'plus', 'cross', 'both'}
        Which polarization to plot
    fftlength : float
        FFT window length in seconds
    title : str, optional
        Plot title (auto-generated if None)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure with spectrogram
    """
    if not isinstance(domain, TimeDomain):
        raise ValueError("Spectrogram requires TimeDomain. Convert waveform to time domain first.")

    h_plus_ts, h_cross_ts = polarization_to_gwpy_timeseries(polarization, domain)

    if title is None:
        title = f"Spectrogram ({polarization_type})"

    if polarization_type == "both":
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("h<sub>+</sub>", "h<sub>x</sub>"),
            horizontal_spacing=0.15,
        )

        spec_plus = h_plus_ts.spectrogram(fftlength=fftlength) ** 0.5
        spec_cross = h_cross_ts.spectrogram(fftlength=fftlength) ** 0.5

        fig.add_trace(
            go.Heatmap(
                x=spec_plus.times.value,
                y=spec_plus.frequencies.value,
                z=np.log10(spec_plus.value.T + 1e-30),
                colorscale="Viridis",
                colorbar=dict(title="log10(Amplitude)", x=0.45),
                hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>log10(amp)=%{z:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=spec_cross.times.value,
                y=spec_cross.frequencies.value,
                z=np.log10(spec_cross.value.T + 1e-30),
                colorscale="Viridis",
                colorbar=dict(title="log10(Amplitude)", x=1.0),
                hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>log10(amp)=%{z:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)

        fig.update_layout(
            title=title, height=height, width=height * 2, template="plotly_white"
        )

    else:
        ts = h_plus_ts if polarization_type == "plus" else h_cross_ts
        spec = ts.spectrogram(fftlength=fftlength) ** 0.5

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=spec.times.value,
                y=spec.frequencies.value,
                z=np.log10(spec.value.T + 1e-30),
                colorscale="Viridis",
                colorbar=dict(title="log10(Amplitude)"),
                hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>log10(amp)=%{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=height,
            template="plotly_white",
        )

    return fig


def plot_polarization_qtransform(
    polarization: Polarization,
    domain: Domain,
    polarization_type: Literal["plus", "cross", "both"] = "plus",
    qrange: tuple = (4.0, 64.0),
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """
    Q-transform time-frequency representation.

    The Q-transform is optimized for detecting transient signals like
    gravitational wave mergers.

    Parameters
    ----------
    polarization : Polarization
        Polarization object
    domain : Domain
        Domain specification (must be TimeDomain)
    polarization_type : {'plus', 'cross', 'both'}
        Which polarization to plot
    qrange : tuple of float
        (Q_min, Q_max) for Q-transform
    title : str, optional
        Plot title (auto-generated if None)
    height : int
        Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure with Q-transform
    """
    if not isinstance(domain, TimeDomain):
        raise ValueError("Q-transform requires TimeDomain. Convert waveform to time domain first.")

    h_plus_ts, h_cross_ts = polarization_to_gwpy_timeseries(polarization, domain)

    if title is None:
        title = f"Q-transform ({polarization_type})"

    if polarization_type == "both":
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("h<sub>+</sub>", "h<sub>x</sub>"),
            horizontal_spacing=0.15,
        )

        qt_plus = h_plus_ts.q_transform(qrange=qrange)
        qt_cross = h_cross_ts.q_transform(qrange=qrange)

        fig.add_trace(
            go.Heatmap(
                x=qt_plus.times.value,
                y=qt_plus.frequencies.value,
                z=qt_plus.value.T,
                colorscale="Viridis",
                colorbar=dict(title="Energy", x=0.45),
                hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>energy=%{z:.2e}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=qt_cross.times.value,
                y=qt_cross.frequencies.value,
                z=qt_cross.value.T,
                colorscale="Viridis",
                colorbar=dict(title="Energy", x=1.0),
                hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>energy=%{z:.2e}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)

        fig.update_layout(
            title=title, height=height, width=height * 2, template="plotly_white"
        )

    else:
        ts = h_plus_ts if polarization_type == "plus" else h_cross_ts
        qt = ts.q_transform(qrange=qrange)

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=qt.times.value,
                y=qt.frequencies.value,
                z=qt.value.T,
                colorscale="Viridis",
                colorbar=dict(title="Energy"),
                hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>energy=%{z:.2e}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            yaxis_type="log",
            height=height,
            template="plotly_white",
        )

    return fig
