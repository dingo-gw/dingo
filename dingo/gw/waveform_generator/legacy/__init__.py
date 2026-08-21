"""
Deprecated legacy WFG API surface.

These wrappers exist purely to give a soft-landing to external code
written against the pre-refactor waveform-generator interface. Every
name here delegates to the natural (typed) API in
``dingo.gw.waveform_generator`` and emits a ``DeprecationWarning``.

New code should import from ``dingo.gw.waveform_generator`` directly.
"""

from .wrappers import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
    sum_contributions_m,
    generate_waveforms_parallel,
)

__all__ = [
    "WaveformGenerator",
    "NewInterfaceWaveformGenerator",
    "sum_contributions_m",
    "generate_waveforms_parallel",
]
