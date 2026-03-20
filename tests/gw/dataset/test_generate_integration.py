"""Integration tests for new-style dataset generation.

Ported from dingo-waveform tests/test_dataset_generation.py
(TestGenerateWaveformDataset section). Uses RandomApproximant.
"""

import numpy as np
import pandas as pd
import pytest

from dingo.gw.dataset.dataset_settings import DatasetSettings
from dingo.gw.dataset.new_generate import new_generate_waveform_dataset
from dingo.gw.dataset.new_waveform_dataset import NewWaveformDataset
from dingo.gw.dataset.waveform_generator_settings import WaveformGeneratorSettings
from dingo.gw.domains import DomainParameters
from dingo.gw.prior import IntrinsicPriors


@pytest.fixture
def basic_settings():
    """Create basic dataset settings using RandomApproximant."""
    return DatasetSettings(
        domain=DomainParameters(
            type="UniformFrequencyDomain",
            f_min=20.0,
            f_max=256.0,
            delta_f=1.0,
        ),
        waveform_generator=WaveformGeneratorSettings(
            approximant="RandomApproximant",
            f_ref=20.0,
        ),
        intrinsic_prior=IntrinsicPriors(
            mass_1="bilby.core.prior.Uniform(minimum=20.0, maximum=50.0)",
            mass_2="bilby.core.prior.Uniform(minimum=10.0, maximum=30.0)",
            luminosity_distance="bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)",
            phase='bilby.core.prior.Uniform(minimum=0.0, maximum=6.28, boundary="periodic")',
        ),
        num_samples=5,
    )


class TestGenerateWaveformDataset:
    """Tests for new_generate_waveform_dataset function."""

    def test_generate_sequential(self, basic_settings):
        dataset = new_generate_waveform_dataset(basic_settings, num_processes=1)

        assert isinstance(dataset, NewWaveformDataset)
        assert len(dataset) <= basic_settings.num_samples
        assert hasattr(dataset.polarizations, "h_plus")
        assert hasattr(dataset.polarizations, "h_cross")
        assert dataset.polarizations.h_plus.shape[0] == len(dataset)
        assert dataset.polarizations.h_cross.shape[0] == len(dataset)

        # Check that waveforms are not all zeros
        assert np.abs(dataset.polarizations.h_plus).max() > 0.0
        assert np.abs(dataset.polarizations.h_cross).max() > 0.0

    def test_waveform_shapes(self, basic_settings):
        dataset = new_generate_waveform_dataset(basic_settings, num_processes=1)

        expected_length = int(
            basic_settings.domain.f_max / basic_settings.domain.delta_f
        ) + 1

        assert dataset.polarizations.h_plus.shape == (
            len(dataset),
            expected_length,
        )
        assert dataset.polarizations.h_cross.shape == (
            len(dataset),
            expected_length,
        )

    def test_parameter_columns(self, basic_settings):
        dataset = new_generate_waveform_dataset(basic_settings, num_processes=1)

        # Should have the parameters we sampled
        core_params = {"mass_1", "mass_2", "luminosity_distance", "phase"}
        actual_params = set(dataset.parameters.columns)
        assert core_params.issubset(actual_params)

    def test_settings_stored(self, basic_settings):
        dataset = new_generate_waveform_dataset(basic_settings, num_processes=1)

        assert dataset.settings is not None
        assert isinstance(dataset.settings, DatasetSettings)

    def test_round_trip_save_load(self, basic_settings, tmp_path):
        dataset = new_generate_waveform_dataset(basic_settings, num_processes=1)

        save_path = tmp_path / "test_dataset.hdf5"
        dataset.save(save_path)
        loaded = NewWaveformDataset.load(save_path)

        assert len(loaded) == len(dataset)
        assert np.allclose(
            loaded.polarizations.h_plus, dataset.polarizations.h_plus
        )
        assert np.allclose(
            loaded.polarizations.h_cross, dataset.polarizations.h_cross
        )

    def test_reproducibility(self, basic_settings):
        """Two datasets with same settings should produce different waveforms (random prior)."""
        dataset1 = new_generate_waveform_dataset(basic_settings, num_processes=1)
        dataset2 = new_generate_waveform_dataset(basic_settings, num_processes=1)

        # Parameters should be different (random sampling)
        assert not np.allclose(
            dataset1.parameters["mass_1"].values,
            dataset2.parameters["mass_1"].values,
        )
