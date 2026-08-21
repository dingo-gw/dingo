"""
Adapter helpers between the legacy dict-based WFG surface and the new
dataclass-based one.

These functions convert:
- legacy ``wfg_kwargs`` (dict) → new ``build_waveform_generator`` schema,
  dropping the historic ``new_interface`` flag and routing unknown keys
  into ``extra_kwargs`` for gwsignal passthrough.
- legacy per-waveform ``theta`` dict → ``BBHWaveformParameters``, ignoring
  keys the dataclass doesn't accept.
- ``Polarization`` (new) → ``{"h_plus": arr, "h_cross": arr}`` (legacy).

Kept public so `dingo.gw.injection` (which straddles both surfaces) and
the legacy wrappers in ``dingo.gw.waveform_generator.legacy`` share the
same conversion logic.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

import numpy as np

from .polarizations import Polarization
from .waveform_parameters import BBHWaveformParameters


_KNOWN_WFG_KEYS = {
    "approximant",
    "f_ref",
    "f_start",
    "mode_list",
    "transform",
    "spin_conversion_phase",
}


def translate_wfg_kwargs(wfg_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Translate legacy WFG kwargs to ``build_waveform_generator`` schema.

    - Drops the legacy ``new_interface`` flag (the factory dispatches by
      approximant).
    - Routes unknown keys (e.g. ``lmax_nyquist``, ``postadiabatic``,
      ``enable_antisymmetric_modes``) into ``extra_kwargs`` for gwsignal
      passthrough.
    """
    result: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in wfg_kwargs.items():
        if k == "new_interface":
            continue
        if k in _KNOWN_WFG_KEYS:
            result[k] = v
        else:
            extra[k] = v
    if extra:
        result["extra_kwargs"] = extra
    return result


_BBH_FIELDS = {f.name for f in fields(BBHWaveformParameters)}


def theta_to_bbh_params(theta_intrinsic: Dict[str, float]) -> BBHWaveformParameters:
    """Build a ``BBHWaveformParameters`` from a legacy theta dict, ignoring
    unknown keys."""
    return BBHWaveformParameters(
        **{k: v for k, v in theta_intrinsic.items() if k in _BBH_FIELDS}
    )


def polarization_to_dict(pol: Polarization) -> Dict[str, np.ndarray]:
    """Convert a ``Polarization`` into the historic ``{h_plus, h_cross}`` dict."""
    return {"h_plus": pol.h_plus, "h_cross": pol.h_cross}
