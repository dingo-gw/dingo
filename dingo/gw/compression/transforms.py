"""Transform pipeline for waveform compression and preprocessing.

This module provides transforms for dataset-level operations like SVD compression
and whitening. These are distinct from the inference-time transforms in
dingo.gw.transforms which handle detector projection, noise injection, etc.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np

from dingo.gw.domains import Domain
from .svd import SVDBasis

_logger = logging.getLogger(__name__)


class Transform(ABC):
    """Abstract base class for waveform compression transforms."""

    @abstractmethod
    def __call__(
        self, polarizations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        pass


class ComposeTransforms:
    """
    Compose multiple transforms into a pipeline.

    Parameters
    ----------
    transforms
        List of Transform instances to apply in sequence.
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(
        self, polarizations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        result = polarizations
        for transform in self.transforms:
            result = transform(result)
        return result

    def __repr__(self) -> str:
        transform_strs = [f"  {t.__class__.__name__}" for t in self.transforms]
        return "ComposeTransforms([\n" + ",\n".join(transform_strs) + "\n])"


class ApplySVD(Transform):
    """
    Transform that applies SVD compression or decompression to waveforms.

    Parameters
    ----------
    svd_basis
        SVDBasis instance with trained basis.
    inverse
        If False, applies compression (default). If True, applies decompression.
    """

    def __init__(self, svd_basis: SVDBasis, inverse: bool = False):
        if svd_basis.V is None:
            raise ValueError("SVD basis has not been trained")
        self.svd_basis = svd_basis
        self.inverse = inverse

    def __call__(
        self, polarizations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        func = (
            self.svd_basis.decompress if self.inverse else self.svd_basis.compress
        )
        return {key: func(value) for key, value in polarizations.items()}

    def __repr__(self) -> str:
        mode = "decompress" if self.inverse else "compress"
        return f"ApplySVD(n_components={self.svd_basis.n_components}, mode={mode})"


class WhitenAndUnwhiten(Transform):
    """
    Transform that whitens or unwhitens waveforms using a fixed ASD.

    Whitening: h_whitened(f) = h(f) / ASD(f)
    Unwhitening: h(f) = h_whitened(f) * ASD(f)

    Parameters
    ----------
    domain
        Frequency domain specification.
    asd_file
        Path to HDF5 file containing the ASD array.
    inverse
        If False (default), apply whitening. If True, apply unwhitening.
    """

    def __init__(
        self,
        domain: Domain,
        asd_file: Union[str, Path],
        inverse: bool = False,
    ):
        self.domain = domain
        self.asd_file = Path(asd_file)
        self.inverse = inverse

        if not self.asd_file.exists():
            raise FileNotFoundError(f"ASD file not found: {self.asd_file}")

        self.asd = self._load_asd()

        _logger.info(
            f"Loaded ASD from {self.asd_file} "
            f"(mode: {'unwhiten' if inverse else 'whiten'})"
        )

    def _load_asd(self) -> np.ndarray:
        """Load ASD array from HDF5 file."""
        with h5py.File(self.asd_file, "r") as f:
            possible_keys = ["asd", "ASD", "asds/H1", "asds/L1"]

            asd = None
            for key in possible_keys:
                if key in f:
                    asd = f[key][:]
                    _logger.debug(f"Loaded ASD from key '{key}'")
                    break

            if asd is None:
                available_keys = list(f.keys())
                raise KeyError(
                    f"Could not find ASD data in {self.asd_file}.\n"
                    f"Tried keys: {possible_keys}\n"
                    f"Available keys in file: {available_keys}"
                )

        expected_length = len(self.domain)
        if len(asd) != expected_length:
            raise ValueError(
                f"ASD length ({len(asd)}) does not match domain length ({expected_length})"
            )

        return asd

    def __call__(
        self, polarizations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if self.inverse:
            return {key: value * self.asd for key, value in polarizations.items()}
        else:
            return {key: value / self.asd for key, value in polarizations.items()}

    def __repr__(self) -> str:
        mode = "unwhiten" if self.inverse else "whiten"
        return f"WhitenAndUnwhiten(asd_file={self.asd_file.name}, mode={mode})"
