"""
Interactive plotting module for dingo waveforms.

Provides plotting functions for both regular polarizations
(from generate_hplus_hcross) and mode-separated polarizations
(from generate_hplus_hcross_m).

All plotting functions return plotly Figure objects for interactive
visualization.
"""

# Polarization plots (for generate_hplus_hcross output)
from .polarization_plots import (
    plot_polarizations_time,
    plot_polarizations_frequency,
    plot_polarization_spectrogram,
    plot_polarization_qtransform,
)

# Mode plots (for generate_hplus_hcross_m output)
from .mode_plots import (
    plot_mode_amplitudes,
    plot_individual_modes,
    plot_mode_comparison,
    plot_modes_grid,
    plot_mode_reconstruction,
)

# Converters (for advanced users)
from .converters import (
    polarization_to_gwpy_timeseries,
    polarization_to_gwpy_frequencyseries,
    modes_to_gwpy_dict_timeseries,
    modes_to_gwpy_dict_frequencyseries,
    gwpy_to_polarization,
)

__all__ = [
    # Polarization plots
    "plot_polarizations_time",
    "plot_polarizations_frequency",
    "plot_polarization_spectrogram",
    "plot_polarization_qtransform",
    # Mode plots
    "plot_mode_amplitudes",
    "plot_individual_modes",
    "plot_mode_comparison",
    "plot_modes_grid",
    "plot_mode_reconstruction",
    # Converters
    "polarization_to_gwpy_timeseries",
    "polarization_to_gwpy_frequencyseries",
    "modes_to_gwpy_dict_timeseries",
    "modes_to_gwpy_dict_frequencyseries",
    "gwpy_to_polarization",
]
