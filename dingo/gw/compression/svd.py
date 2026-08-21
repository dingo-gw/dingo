"""SVD-based compression for waveform datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import h5py
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.utils.extmath import randomized_svd

from dingo.gw.types import (
    SVDBasisMatrix,
    SVDBasisMatrixH,
    SingularValues,
    SVDCoefficients,
    WaveformTrainingData,
)

_logger = logging.getLogger(__name__)


@dataclass
class SVDBasis:
    """
    SVD basis for compressing waveform datasets.

    Uses singular value decomposition to create a compressed representation
    of waveforms. The decomposition is:
        training_data = U @ diag(s) @ Vh

    where U and Vh are unitary matrices.

    Attributes
    ----------
    V : SVDBasisMatrix | None
        Right singular vectors (basis), shape (n_features, n_components).
    Vh : SVDBasisMatrixH | None
        Hermitian conjugate of V, shape (n_components, n_features).
    s : SingularValues | None
        Singular values, shape (n_components,).
    n_components : int | None
        Number of basis elements kept.
    """

    V: Optional[SVDBasisMatrix] = None
    Vh: Optional[SVDBasisMatrixH] = None
    s: Optional[SingularValues] = None
    n_components: Optional[int] = None

    def generate_basis(
        self,
        training_data: WaveformTrainingData,
        n_components: int = 0,
        method: Literal["scipy", "randomized"] = "scipy",
    ) -> None:
        """
        Generate SVD basis from training data.

        Parameters
        ----------
        training_data
            Array of waveform data with shape (n_samples, n_features).
        n_components
            Number of basis elements to keep. If 0, keeps all components.
        method
            SVD method: "scipy" (full, more accurate) or "randomized" (faster).
        """
        if training_data.size == 0:
            raise ValueError("training_data cannot be empty")

        if method == "randomized":
            if n_components == 0:
                n_components = min(training_data.shape)

            _logger.info(
                f"Generating randomized SVD basis with {n_components} components..."
            )

            try:
                U, s, Vh = randomized_svd(
                    training_data,
                    n_components,
                    random_state=0,
                    power_iteration_normalizer="QR",
                )
            except ValueError as e:
                if "complex" in str(e).lower():
                    raise ValueError(
                        "randomized_svd failed with complex data. "
                        "This may be due to scikit-learn >= 1.2. "
                        "Either use method='scipy' or downgrade scikit-learn to 1.1.3."
                    ) from e
                raise

            self.Vh = Vh.astype(np.complex128)
            self.V = self.Vh.T.conj()
            self.n_components = n_components
            self.s = s

        elif method == "scipy":
            _logger.info("Generating full SVD basis using scipy...")

            U, s, Vh = scipy.linalg.svd(training_data, full_matrices=False)
            V = Vh.T.conj()

            if n_components == 0 or n_components > len(V):
                self.V = V
                self.Vh = Vh
            else:
                self.V = V[:, :n_components]
                self.Vh = Vh[:n_components, :]

            self.n_components = self.Vh.shape[0]
            self.s = s[: self.n_components]

        else:
            raise ValueError(
                f"Invalid method '{method}'. Use 'scipy' or 'randomized'."
            )

        _logger.info(f"SVD basis generated with {self.n_components} components")

    def compress(self, data: np.ndarray) -> SVDCoefficients:
        """
        Compress data using the SVD basis.

        Parameters
        ----------
        data
            Waveform data to compress. Shape (n_features,) or (n_samples, n_features).

        Returns
        -------
        Compressed coefficients.
        """
        if self.V is None:
            raise ValueError(
                "SVD basis has not been generated. Call generate_basis() first."
            )
        return data @ self.V

    def decompress(self, coefficients: SVDCoefficients) -> np.ndarray:
        """
        Decompress coefficients back to waveform data.

        Parameters
        ----------
        coefficients
            SVD coefficients. Shape (n_components,) or (n_samples, n_components).

        Returns
        -------
        Reconstructed waveform data.
        """
        if self.Vh is None:
            raise ValueError(
                "SVD basis has not been generated. Call generate_basis() first."
            )
        return coefficients @ self.Vh

    def compute_mismatches(
        self,
        data: WaveformTrainingData,
        parameters: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute reconstruction mismatches for validation.

        Mismatch is defined as: 1 - <d1, d2> / (||d1|| ||d2||)

        Parameters
        ----------
        data
            Validation data, shape (n_samples, n_features).
        parameters
            Optional parameters DataFrame for labeling results.

        Returns
        -------
        DataFrame with mismatch values.
        """
        if self.V is None or self.Vh is None:
            raise ValueError("SVD basis has not been generated.")

        _logger.info(f"Computing mismatches for {len(data)} waveforms...")

        if parameters is not None:
            if len(data) != len(parameters):
                raise ValueError(
                    f"Data length ({len(data)}) does not match "
                    f"parameters length ({len(parameters)})"
                )
            results = parameters.copy()
        else:
            results = pd.DataFrame()

        mismatches = np.empty(len(data))
        for i, waveform in enumerate(data):
            compressed = self.compress(waveform)
            reconstructed = self.decompress(compressed)
            norm1 = np.sqrt(np.sum(np.abs(waveform) ** 2))
            norm2 = np.sqrt(np.sum(np.abs(reconstructed) ** 2))
            inner = np.sum(waveform.conj() * reconstructed).real
            mismatches[i] = 1.0 - inner / (norm1 * norm2)

        results[f"mismatch_n={self.n_components}"] = mismatches

        _logger.info(f"Mismatch statistics (n={self.n_components}):")
        _logger.info(f"  Mean: {np.mean(mismatches):.2e}")
        _logger.info(f"  Std: {np.std(mismatches):.2e}")
        _logger.info(f"  Max: {np.max(mismatches):.2e}")
        _logger.info(f"  Median: {np.median(mismatches):.2e}")
        _logger.info(f"  99th percentile: {np.percentile(mismatches, 99):.2e}")

        return results

    def save(self, file_path: Union[str, Path]) -> None:
        """Save SVD basis to HDF5 file."""
        if self.V is None:
            raise ValueError(
                "SVD basis has not been generated. Nothing to save."
            )

        file_path = Path(file_path)
        _logger.info(f"Saving SVD basis to {file_path}")

        with h5py.File(file_path, "w") as f:
            f.create_dataset("V", data=self.V, compression="gzip")
            f.create_dataset("s", data=self.s, compression="gzip")
            f.attrs["n_components"] = self.n_components

        _logger.info(f"SVD basis saved ({self.n_components} components)")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> SVDBasis:
        """Load SVD basis from HDF5 file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SVD basis file not found: {file_path}")

        _logger.info(f"Loading SVD basis from {file_path}")

        with h5py.File(file_path, "r") as f:
            if "V" not in f:
                raise KeyError(
                    f"File {file_path} does not contain SVD V matrix"
                )

            V = f["V"][:]
            s = f["s"][:] if "s" in f else None
            n_components = f.attrs.get("n_components", V.shape[1])

        basis = cls(V=V, Vh=V.T.conj(), s=s, n_components=n_components)

        _logger.info(f"SVD basis loaded ({n_components} components)")
        return basis

    def to_dict(self) -> Dict:
        """Convert SVD basis to dictionary for storage."""
        if self.V is None:
            return {}
        return {
            "V": self.V,
            "s": self.s,
            "n_components": self.n_components,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> SVDBasis:
        """Create SVD basis from dictionary."""
        if not data or "V" not in data:
            return cls()

        V = data["V"]
        return cls(
            V=V,
            Vh=V.T.conj(),
            s=data.get("s"),
            n_components=data.get("n_components", V.shape[1]),
        )
