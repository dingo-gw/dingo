"""
Binning subpackage for adaptive frequency-domain binning.

This subpackage provides utilities for multi-banded frequency domain binning
with dyadic spacing and decimation operations.
"""

from .adaptive_binning import (
    Band,
    BinningParameters,
    compute_adaptive_binning,
    compile_binning_from_bands,
    decimate,
    decimate_uniform,
    plan_bands,
)

__all__ = [
    "Band",
    "BinningParameters",
    "compute_adaptive_binning",
    "compile_binning_from_bands",
    "decimate",
    "decimate_uniform",
    "plan_bands",
]
