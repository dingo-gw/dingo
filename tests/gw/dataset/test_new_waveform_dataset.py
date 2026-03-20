"""Tests for NewWaveformDataset.

Ported from dingo-waveform tests/test_dataset_generation.py (dataset parts).
"""

import numpy as np
import pandas as pd
import pytest

from dingo.gw.waveform_generator.polarizations import BatchPolarizations
from dingo.gw.dataset.new_waveform_dataset import NewWaveformDataset


class TestNewWaveformDataset:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_waveforms = 10
        n_freq_bins = 100
        parameters = pd.DataFrame(
            {
                "mass_1": np.random.uniform(10, 50, n_waveforms),
                "mass_2": np.random.uniform(5, 30, n_waveforms),
            }
        )
        polarizations = BatchPolarizations(
            h_plus=np.random.randn(n_waveforms, n_freq_bins)
            + 1j * np.random.randn(n_waveforms, n_freq_bins),
            h_cross=np.random.randn(n_waveforms, n_freq_bins)
            + 1j * np.random.randn(n_waveforms, n_freq_bins),
        )
        return parameters, polarizations

    def test_create_from_batch_polarizations(self, sample_data):
        parameters, polarizations = sample_data
        dataset = NewWaveformDataset(parameters=parameters, polarizations=polarizations)

        assert len(dataset) == 10
        assert dataset.parameters is parameters
        assert dataset.polarizations is polarizations
        assert dataset.settings is None
        assert dataset.svd_basis is None

    def test_create_from_dict(self, sample_data):
        parameters, polarizations = sample_data
        pol_dict = {
            "h_plus": polarizations.h_plus,
            "h_cross": polarizations.h_cross,
        }
        dataset = NewWaveformDataset(parameters=parameters, polarizations=pol_dict)

        assert len(dataset) == 10
        assert isinstance(dataset.polarizations, BatchPolarizations)
        assert np.array_equal(dataset.polarizations.h_plus, polarizations.h_plus)

    def test_validation_mismatch(self, sample_data):
        parameters, polarizations = sample_data
        # Remove some parameters to create mismatch
        parameters_short = parameters.iloc[:5]
        with pytest.raises(ValueError, match="Mismatch"):
            NewWaveformDataset(
                parameters=parameters_short, polarizations=polarizations
            )

    def test_repr(self, sample_data):
        parameters, polarizations = sample_data
        dataset = NewWaveformDataset(parameters=parameters, polarizations=polarizations)
        repr_str = repr(dataset)
        assert "NewWaveformDataset" in repr_str
        assert "num_waveforms=10" in repr_str

    def test_save_and_load(self, sample_data, tmp_path):
        parameters, polarizations = sample_data
        dataset = NewWaveformDataset(parameters=parameters, polarizations=polarizations)

        save_path = tmp_path / "test_dataset.hdf5"
        dataset.save(save_path)
        assert save_path.exists()

        loaded = NewWaveformDataset.load(save_path)
        assert len(loaded) == len(dataset)
        assert np.allclose(loaded.polarizations.h_plus, dataset.polarizations.h_plus)
        assert np.allclose(loaded.polarizations.h_cross, dataset.polarizations.h_cross)
        assert list(loaded.parameters.columns) == list(dataset.parameters.columns)
        assert np.allclose(loaded.parameters["mass_1"].values, dataset.parameters["mass_1"].values)

    def test_save_and_load_with_settings(self, sample_data, tmp_path):
        parameters, polarizations = sample_data
        settings_dict = {
            "num_samples": 10,
            "domain": {"type": "FrequencyDomain", "f_min": 20.0, "f_max": 1024.0},
        }
        dataset = NewWaveformDataset(
            parameters=parameters,
            polarizations=polarizations,
            settings=settings_dict,
        )

        save_path = tmp_path / "test_dataset_settings.hdf5"
        dataset.save(save_path)

        loaded = NewWaveformDataset.load(save_path)
        assert loaded.settings is not None
        assert loaded.settings["num_samples"] == 10

    def test_save_and_load_with_svd_basis(self, sample_data, tmp_path):
        from dingo.gw.compression.svd import SVDBasis

        parameters, polarizations = sample_data
        basis = SVDBasis()
        basis.generate_basis(polarizations.h_plus, n_components=5, method="scipy")

        dataset = NewWaveformDataset(
            parameters=parameters,
            polarizations=polarizations,
            svd_basis=basis,
        )

        save_path = tmp_path / "test_dataset_svd.hdf5"
        dataset.save(save_path)

        loaded = NewWaveformDataset.load(save_path)
        assert loaded.svd_basis is not None
        assert loaded.svd_basis.n_components == 5
        assert np.allclose(loaded.svd_basis.V, basis.V)

    def test_settings_as_dict(self, sample_data):
        parameters, polarizations = sample_data
        settings = {"key": "value"}
        dataset = NewWaveformDataset(
            parameters=parameters, polarizations=polarizations, settings=settings
        )
        assert dataset.settings == {"key": "value"}
