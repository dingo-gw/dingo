from .utils import merge_datasets
from .compression_settings import CompressionSettings, SVDSettings
from .dataset_settings import DatasetSettings
from .generation_types import WaveformGeneratorConfig, WaveformResult
from .waveform_generator_settings import WaveformGeneratorSettings
from .dataset import WaveformDataset
from .generate import (
    generate_waveform_dataset,
    generate_parameters_and_polarizations,
    generate_waveforms_sequential,
    generate_waveforms_parallel,
    generate_waveforms_parallel_optimized,
    build_compression_transforms,
    train_svd_basis,
)

# The legacy dict-based WaveformDataset container remains importable at its
# fully qualified path (`dingo.gw.dataset.waveform_dataset`) for the training
# pipeline and legacy callers. Top-level `dingo.gw.dataset.WaveformDataset`
# is the new dataclass-based container.
