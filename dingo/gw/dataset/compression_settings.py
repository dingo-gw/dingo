"""Settings for dataset compression (SVD and whitening)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class SVDSettings:
    """
    Configuration for SVD compression of waveforms.

    Attributes
    ----------
    size
        Number of SVD basis elements to keep. Set to 0 for no truncation.
    num_training_samples
        Number of waveforms to use for training the SVD basis.
    num_validation_samples
        Number of waveforms to use for validating the SVD basis.
    file
        Optional path to pre-computed SVD basis file.
    """

    size: int
    num_training_samples: int
    num_validation_samples: int = 0
    file: Optional[Union[str, Path]] = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"SVD size must be non-negative, got {self.size}")

        if self.num_training_samples <= 0:
            raise ValueError(
                f"num_training_samples must be positive, got {self.num_training_samples}"
            )

        if self.num_validation_samples < 0:
            raise ValueError(
                f"num_validation_samples must be non-negative, got {self.num_validation_samples}"
            )

        if isinstance(self.file, str):
            self.file = Path(self.file)


@dataclass
class CompressionSettings:
    """
    Configuration for waveform dataset compression.

    Compression can include SVD projection and/or whitening with a fixed ASD.
    """

    svd: Optional[SVDSettings] = None
    whitening: Optional[Union[str, Path]] = None

    def __post_init__(self):
        if isinstance(self.whitening, str):
            self.whitening = Path(self.whitening)

        if self.svd is None and self.whitening is None:
            raise ValueError(
                "CompressionSettings must specify at least one of: svd, whitening"
            )
