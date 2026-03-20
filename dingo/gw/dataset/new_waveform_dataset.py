"""New-style waveform dataset class for storing and managing generated waveforms.

This is the ported version from dingo-waveform. The existing WaveformDataset
in waveform_dataset.py (which inherits from DingoDataset) is preserved for
backward compatibility.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd

from dingo.gw.compression.svd import SVDBasis
from dingo.gw.waveform_generator.polarizations import BatchPolarizations
from .dataset_settings import DatasetSettings

_logger = logging.getLogger(__name__)


class NewWaveformDataset:
    """
    Container for waveform parameters and polarizations.

    Attributes
    ----------
    parameters : pd.DataFrame
        DataFrame containing waveform parameters (one row per waveform).
    polarizations : BatchPolarizations
        Dataclass containing 'h_plus' and 'h_cross' arrays.
    settings : Optional[DatasetSettings]
        Settings used to generate the dataset.
    svd_basis : Optional[SVDBasis]
        SVD basis if dataset was compressed.
    """

    def __init__(
        self,
        parameters: pd.DataFrame,
        polarizations: Union[BatchPolarizations, Dict[str, np.ndarray]],
        settings: Optional[Union[DatasetSettings, Dict]] = None,
        svd_basis: Optional[SVDBasis] = None,
    ):
        self.parameters = parameters

        if isinstance(polarizations, dict):
            self.polarizations = BatchPolarizations(
                h_plus=polarizations["h_plus"],
                h_cross=polarizations["h_cross"],
            )
        else:
            self.polarizations = polarizations

        if settings is None:
            self.settings = None
        elif isinstance(settings, dict):
            self.settings = settings
            _logger.debug(
                "Received settings as dict; consider using DatasetSettings dataclass"
            )
        else:
            self.settings = settings

        self.svd_basis = svd_basis
        self._validate()

    def _validate(self) -> None:
        num_params = len(self.parameters)
        num_polarizations = len(self.polarizations)

        if num_polarizations != num_params:
            raise ValueError(
                f"Mismatch: {num_params} parameter rows but "
                f"{num_polarizations} waveforms in polarizations"
            )

        _logger.debug(f"Dataset validated: {num_params} waveforms")

    def __len__(self) -> int:
        return len(self.parameters)

    def __repr__(self) -> str:
        return (
            f"NewWaveformDataset(num_waveforms={len(self)}, "
            f"num_parameters={len(self.parameters.columns)}, "
            f"waveform_length={self.polarizations.num_frequency_bins})"
        )

    def save(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        _logger.info(f"Saving dataset to {file_path}")

        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "h_plus", data=self.polarizations.h_plus, compression="gzip"
            )
            f.create_dataset(
                "h_cross", data=self.polarizations.h_cross, compression="gzip"
            )

            params_group = f.create_group("parameters")
            for col in self.parameters.columns:
                params_group.create_dataset(
                    col, data=self.parameters[col].values
                )

            settings_group = f.create_group("settings")
            if self.settings is not None:
                if isinstance(self.settings, DatasetSettings):
                    settings_dict = self.settings.to_dict()
                else:
                    settings_dict = self.settings
                self._save_dict_to_group(settings_group, settings_dict)

            if self.svd_basis is not None:
                _logger.info("Saving SVD basis...")
                svd_group = f.create_group("svd")
                svd_dict = self.svd_basis.to_dict()
                for key, value in svd_dict.items():
                    if value is not None:
                        if isinstance(value, np.ndarray) and value.size > 1:
                            svd_group.create_dataset(
                                key, data=value, compression="gzip"
                            )
                        else:
                            svd_group.create_dataset(key, data=value)

        _logger.info(f"Dataset saved successfully ({len(self)} waveforms)")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "NewWaveformDataset":
        file_path = Path(file_path)
        _logger.info(f"Loading dataset from {file_path}")

        with h5py.File(file_path, "r") as f:
            polarizations = BatchPolarizations(
                h_plus=f["h_plus"][:],
                h_cross=f["h_cross"][:],
            )

            params_group = f["parameters"]
            param_dict = {
                key: params_group[key][:] for key in params_group.keys()
            }
            parameters = pd.DataFrame(param_dict)

            settings = (
                cls._load_dict_from_group(f["settings"])
                if "settings" in f
                else None
            )

            svd_basis = None
            if "svd" in f:
                _logger.info("Loading SVD basis...")
                svd_dict = {}
                for key in f["svd"].keys():
                    svd_dict[key] = f["svd"][key][()]
                svd_basis = SVDBasis.from_dict(svd_dict)

        _logger.info(
            f"Dataset loaded successfully ({len(parameters)} waveforms)"
        )
        return cls(parameters, polarizations, settings, svd_basis)

    @staticmethod
    def _save_dict_to_group(group: h5py.Group, data: Dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                NewWaveformDataset._save_dict_to_group(subgroup, value)
            elif value is None:
                ds = group.create_dataset(key, data=h5py.Empty("f"))
                ds.attrs["is_none"] = True
            elif isinstance(value, (list, tuple)):
                group.create_dataset(key, data=np.array(value))
            elif isinstance(value, (str, int, float, bool, np.ndarray)):
                group.create_dataset(key, data=value)
            else:
                _logger.warning(
                    f"Converting unsupported type {type(value)} to string for key '{key}'"
                )
                group.create_dataset(key, data=str(value))

    @staticmethod
    def _load_dict_from_group(group: h5py.Group) -> Dict:
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = NewWaveformDataset._load_dict_from_group(item)
            elif isinstance(item, h5py.Dataset):
                if item.attrs.get("is_none", False):
                    result[key] = None
                else:
                    data = item[()]
                    if isinstance(data, bytes):
                        result[key] = data.decode("utf-8")
                    else:
                        result[key] = data
        return result
