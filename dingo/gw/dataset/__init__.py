from .generate_dataset import generate_dataset, generate_parameters_and_polarizations
from .utils import merge_datasets
from .waveform_dataset import *

# New-style API (ported from dingo-waveform)
from .compression_settings import CompressionSettings, SVDSettings
from .dataset_settings import DatasetSettings
from .generation_types import WaveformGeneratorConfig, WaveformResult
from .waveform_generator_settings import WaveformGeneratorSettings
from .new_waveform_dataset import NewWaveformDataset
from .new_generate import (
    new_generate_waveform_dataset,
    new_generate_parameters_and_polarizations,
    generate_waveforms_sequential,
    generate_waveforms_parallel,
    generate_waveforms_parallel_optimized,
    build_compression_transforms,
    train_svd_basis,
)
