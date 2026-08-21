"""
Deprecated shim for the legacy ``dingo.gw.dataset.generate_dataset``
module path.

The historic module was deleted during the WFG refactor. This shim
reintroduces the same names, delegating to the natural API and emitting a
``DeprecationWarning`` on use.

New code should import from ``dingo.gw.dataset`` directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dingo.core.utils.deprecation import deprecated

from .cli import generate_dataset_main as _cli_main
from .dataset_settings import DatasetSettings
from .dataset import WaveformDataset
from .generate import (
    generate_waveform_dataset as _generate_waveform_dataset,
    generate_parameters_and_polarizations as _generate_params_pols,
)
from ..waveform_generator.legacy.wrappers import (
    WaveformGenerator as _LegacyWaveformGenerator,
)


@deprecated(
    "dingo.gw.dataset.generate_dataset.generate_dataset was the legacy entrypoint; "
    "use dingo.gw.dataset.generate_waveform_dataset with a DatasetSettings instead.",
    replacement="dingo.gw.dataset.generate_waveform_dataset",
)
def generate_dataset(settings, num_processes: int) -> WaveformDataset:
    """Legacy dataset generator.

    Accepts the historic settings shape (either a ``DatasetSettings`` instance
    or a plain dict as loaded from YAML) and returns a new-API
    ``WaveformDataset``. Historic callers unwrap ``dataset.parameters`` /
    ``dataset.polarizations`` themselves; both surfaces are preserved on the
    new container.
    """
    if not isinstance(settings, DatasetSettings):
        settings = DatasetSettings.from_dict(settings)
    return _generate_waveform_dataset(settings, num_processes=num_processes)


@deprecated(
    "generate_parameters_and_polarizations over the legacy WFG returns a dict of "
    "stacked arrays; the natural API returns a BatchPolarizations dataclass.",
    replacement="dingo.gw.dataset.generate_parameters_and_polarizations",
)
def generate_parameters_and_polarizations(
    waveform_generator,
    prior,
    num_samples: int,
    num_processes: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Legacy signature: returns ``(parameters_df, {"h_plus": arr, "h_cross": arr})``."""
    inner = (
        waveform_generator._impl
        if isinstance(waveform_generator, _LegacyWaveformGenerator)
        else waveform_generator
    )
    parameters, batch = _generate_params_pols(
        inner, prior, num_samples, num_processes
    )
    return parameters, {"h_plus": batch.h_plus, "h_cross": batch.h_cross}


@deprecated(
    "_generate_dataset_main was the CLI helper on the deleted module.",
    replacement="dingo.gw.dataset.cli.generate_dataset_main",
)
def _generate_dataset_main(settings_file: str, out_file: str, num_processes: int):
    return _cli_main(settings_file, out_file, num_processes)
