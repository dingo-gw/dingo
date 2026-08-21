"""
Legacy WFG wrappers — dict in, dict out — delegating to the natural API.

Each public symbol is decorated with ``@deprecated`` so callers see a
warning steering them to the typed replacement.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dingo.core.utils.deprecation import deprecated

from ..adapters import (
    polarization_to_dict,
    theta_to_bbh_params,
    translate_wfg_kwargs,
)
from ..api import (
    WaveformGenerator as _NewWaveformGenerator,
    build_waveform_generator,
)


@deprecated(
    "Dict-in / dict-out WFG has been replaced by the typed API.",
    replacement=(
        "dingo.gw.waveform_generator.build_waveform_generator + "
        "BBHWaveformParameters"
    ),
)
class WaveformGenerator:
    """Dict-based facade over the typed
    :class:`dingo.gw.waveform_generator.WaveformGenerator`.

    Accepts the historic ``wfg_kwargs`` at construction, ``theta`` dicts
    at generation, and returns ``{"h_plus": ..., "h_cross": ...}``
    dicts. Kept for backward compatibility only.
    """

    def __init__(self, approximant: str, domain, f_ref: float, **wfg_kwargs: Any):
        kwargs = {"approximant": approximant, "f_ref": f_ref, **wfg_kwargs}
        self._impl: _NewWaveformGenerator = build_waveform_generator(
            translate_wfg_kwargs(kwargs), domain=domain
        )

    @property
    def domain(self):
        return self._impl.domain

    @property
    def base_domain(self):
        return self._impl.base_domain

    @property
    def approximant(self):
        return self._impl._waveform_gen_params.approximant

    @property
    def f_ref(self):
        return self._impl._waveform_gen_params.f_ref

    @property
    def f_start(self):
        return self._impl._waveform_gen_params.f_start

    @property
    def spin_conversion_phase(self):
        return self._impl._waveform_gen_params.spin_conversion_phase

    @property
    def transform(self):
        return self._impl.transform

    @transform.setter
    def transform(self, value):
        self._impl.transform = value

    def generate_hplus_hcross(
        self,
        parameters: Dict[str, float],
        catch_waveform_errors: bool = False,
    ) -> Dict[str, np.ndarray]:
        pol = self._impl.generate_hplus_hcross(
            theta_to_bbh_params(parameters),
            catch_waveform_errors=catch_waveform_errors,
        )
        return polarization_to_dict(pol)

    def generate_hplus_hcross_m(
        self, parameters: Dict[str, float]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        pol_m = self._impl.generate_hplus_hcross_m(
            theta_to_bbh_params(parameters)
        )
        return {int(m): polarization_to_dict(p) for m, p in pol_m.items()}


@deprecated(
    "NewInterfaceWaveformGenerator was a gwsignal-dispatch flag; the natural "
    "factory dispatches by approximant.",
    replacement=(
        "dingo.gw.waveform_generator.build_waveform_generator "
        "(GWSignal-backed approximants are dispatched automatically)"
    ),
)
class NewInterfaceWaveformGenerator(WaveformGenerator):
    """Legacy alias — the natural factory dispatches by approximant name,
    so this is functionally equivalent to :class:`WaveformGenerator`
    (dict-based)."""


@deprecated(
    "sum_contributions_m over a per-detector dict-of-dicts is legacy.",
    replacement=(
        "dingo.gw.waveform_generator.sum_contributions_m on a Dict[Mode, Polarization]"
    ),
)
def sum_contributions_m(x_m: Dict[Any, Dict[Any, np.ndarray]], phase_shift: float = 0.0):
    """Sum contributions over m-components on a dict-of-dicts (per-detector or
    per-polarization keys)."""
    keys = next(iter(x_m.values())).keys()
    result: Dict[Any, Any] = {k: 0.0 for k in keys}
    for k in keys:
        for m, x in x_m.items():
            result[k] = result[k] + x[k] * np.exp(-1j * m * phase_shift)
    return result


@deprecated(
    "generate_waveforms_parallel(wfg, parameters, pool=None) is the legacy "
    "signature over a legacy WaveformGenerator.",
    replacement=(
        "dingo.gw.dataset.generate_waveforms_parallel(waveform_generator, "
        "parameters, num_processes)"
    ),
)
def generate_waveforms_parallel(
    waveform_generator: "WaveformGenerator",
    parameters: pd.DataFrame,
    pool: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Legacy parallel generator: dict of stacked ``h_plus``/``h_cross`` arrays.

    ``pool`` is accepted only for signature compatibility; the natural
    generator infers concurrency from ``num_processes`` internally. When a
    ``pool`` is provided we fall back to sequential generation to avoid
    double-parallelism; users on the typed API should pass an integer
    ``num_processes`` instead.
    """
    from dingo.gw.dataset.generate import (
        generate_waveforms_parallel as _new_parallel,
        generate_waveforms_sequential as _new_sequential,
    )

    inner = waveform_generator._impl if isinstance(
        waveform_generator, WaveformGenerator
    ) else waveform_generator

    if pool is None:
        batch = _new_sequential(inner, parameters)
    else:
        batch = _new_parallel(inner, parameters, num_processes=1)

    return {"h_plus": batch.h_plus, "h_cross": batch.h_cross}
