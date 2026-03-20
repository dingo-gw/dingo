"""Integration tests for compression in dataset generation.

Ported from dingo-waveform tests/test_compression_integration.py.
Uses RandomApproximant since LAL approximants are stubs in the new API.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from dingo.gw.compression.svd import SVDBasis
from dingo.gw.compression.transforms import ApplySVD, ComposeTransforms, WhitenAndUnwhiten
from dingo.gw.dataset.compression_settings import CompressionSettings, SVDSettings
from dingo.gw.dataset.new_generate import (
    apply_transforms_to_polarizations,
    build_compression_transforms,
    train_svd_basis,
)
from dingo.gw.dataset.new_waveform_dataset import NewWaveformDataset
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator.new_api import build_waveform_generator
from dingo.gw.waveform_generator.polarizations import BatchPolarizations


@pytest.fixture
def domain():
    return UniformFrequencyDomain(f_min=20.0, f_max=256.0, delta_f=1.0)


@pytest.fixture
def waveform_generator(domain):
    return build_waveform_generator(
        {"approximant": "RandomApproximant", "f_ref": 20.0}, domain
    )


@pytest.fixture
def sample_polarizations(domain):
    """Generate sample polarizations matching domain length."""
    np.random.seed(42)
    n = 30
    length = len(domain)
    return BatchPolarizations(
        h_plus=np.random.randn(n, length) + 1j * np.random.randn(n, length),
        h_cross=np.random.randn(n, length) + 1j * np.random.randn(n, length),
    )


class TestCompressionSettings:
    """Test CompressionSettings dataclass."""

    def test_svd_only(self):
        settings = CompressionSettings(
            svd=SVDSettings(size=50, num_training_samples=10, num_validation_samples=5)
        )
        assert settings.svd.size == 50
        assert settings.whitening is None

    def test_whitening_only(self, tmp_path):
        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=np.ones(100))

        settings = CompressionSettings(whitening=str(asd_path))
        assert settings.whitening == asd_path
        assert settings.svd is None

    def test_both_svd_and_whitening(self, tmp_path):
        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=np.ones(100))

        settings = CompressionSettings(
            svd=SVDSettings(size=50, num_training_samples=10),
            whitening=str(asd_path),
        )
        assert settings.svd is not None
        assert settings.whitening is not None

    def test_error_no_compression(self):
        with pytest.raises(ValueError, match="must specify at least one"):
            CompressionSettings()


class TestSVDCompressionPipeline:
    """Test SVD compression applied to polarizations."""

    def test_svd_compress_polarizations(self, sample_polarizations):
        n_components = 10
        basis = SVDBasis()
        train_data = np.concatenate(
            [sample_polarizations.h_plus[:20], sample_polarizations.h_cross[:20]],
            axis=0,
        )
        basis.generate_basis(train_data, n_components=n_components, method="scipy")

        transforms = ComposeTransforms([ApplySVD(basis, inverse=False)])
        result = apply_transforms_to_polarizations(sample_polarizations, transforms)

        assert result.h_plus.shape == (30, n_components)
        assert result.h_cross.shape == (30, n_components)

    def test_svd_roundtrip(self, sample_polarizations):
        n_components = 20
        basis = SVDBasis()
        train_data = np.concatenate(
            [sample_polarizations.h_plus, sample_polarizations.h_cross], axis=0
        )
        basis.generate_basis(train_data, n_components=n_components, method="scipy")

        compress = ComposeTransforms([ApplySVD(basis, inverse=False)])
        decompress = ComposeTransforms([ApplySVD(basis, inverse=True)])

        compressed = apply_transforms_to_polarizations(sample_polarizations, compress)
        reconstructed = apply_transforms_to_polarizations(compressed, decompress)

        assert reconstructed.h_plus.shape == sample_polarizations.h_plus.shape

    def test_train_svd_basis_integration(self, sample_polarizations):
        parameters = pd.DataFrame(
            {"mass_1": np.random.uniform(10, 50, len(sample_polarizations))}
        )

        basis, n_train, n_val = train_svd_basis(
            sample_polarizations, parameters, size=10, n_train=20
        )

        assert basis.n_components == 10
        assert n_train == 20
        assert n_val == 10

        # Verify basis can compress/decompress
        compressed = basis.compress(sample_polarizations.h_plus)
        assert compressed.shape == (30, 10)
        reconstructed = basis.decompress(compressed)
        assert reconstructed.shape == sample_polarizations.h_plus.shape


class TestWhiteningPipeline:
    """Test whitening transform applied to polarizations."""

    def test_whitening_polarizations(self, domain, sample_polarizations):
        asd = np.ones(len(domain)) * 1e-23

        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            asd_path = Path(f.name)
        try:
            with h5py.File(asd_path, "w") as f:
                f.create_dataset("asd", data=asd)

            transforms = ComposeTransforms(
                [WhitenAndUnwhiten(domain, asd_path, inverse=False)]
            )
            result = apply_transforms_to_polarizations(
                sample_polarizations, transforms
            )

            assert result.h_plus.shape == sample_polarizations.h_plus.shape
            # Whitened by dividing by 1e-23, so values should be much larger
            assert np.abs(result.h_plus).max() > np.abs(sample_polarizations.h_plus).max()
        finally:
            asd_path.unlink(missing_ok=True)


class TestCombinedPipeline:
    """Test whitening + SVD pipeline."""

    def test_whitening_then_svd(self, domain, sample_polarizations):
        asd = np.ones(len(domain)) * 1e-23

        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            asd_path = Path(f.name)
        try:
            with h5py.File(asd_path, "w") as f:
                f.create_dataset("asd", data=asd)

            # First whiten, then train SVD on whitened data
            whiten = WhitenAndUnwhiten(domain, asd_path, inverse=False)
            whitened = apply_transforms_to_polarizations(
                sample_polarizations,
                ComposeTransforms([whiten]),
            )

            # Train SVD on whitened data
            train_data = np.concatenate(
                [whitened.h_plus, whitened.h_cross], axis=0
            )
            basis = SVDBasis()
            basis.generate_basis(train_data, n_components=10, method="scipy")

            # Build combined pipeline
            pipeline = ComposeTransforms(
                [
                    WhitenAndUnwhiten(domain, asd_path, inverse=False),
                    ApplySVD(basis, inverse=False),
                ]
            )

            result = apply_transforms_to_polarizations(
                sample_polarizations, pipeline
            )

            assert result.h_plus.shape == (30, 10)
            assert result.h_cross.shape == (30, 10)
        finally:
            asd_path.unlink(missing_ok=True)


class TestBackwardCompatibility:
    """Test backward compatibility with datasets without compression."""

    def test_load_old_dataset_without_svd(self, tmp_path):
        parameters = pd.DataFrame(
            {"mass_1": [35.0, 36.0], "mass_2": [30.0, 31.0]}
        )
        polarizations = BatchPolarizations(
            h_plus=np.random.randn(2, 100) + 1j * np.random.randn(2, 100),
            h_cross=np.random.randn(2, 100) + 1j * np.random.randn(2, 100),
        )

        dataset = NewWaveformDataset(parameters, polarizations)

        path = tmp_path / "old_dataset.hdf5"
        dataset.save(path)
        loaded = NewWaveformDataset.load(path)

        assert loaded.svd_basis is None
        assert len(loaded) == 2

    def test_dataset_with_svd_basis_roundtrip(self, tmp_path, sample_polarizations):
        np.random.seed(42)
        parameters = pd.DataFrame(
            {"mass_1": np.random.uniform(10, 50, len(sample_polarizations))}
        )

        basis = SVDBasis()
        train_data = np.concatenate(
            [sample_polarizations.h_plus, sample_polarizations.h_cross], axis=0
        )
        basis.generate_basis(train_data, n_components=10, method="scipy")

        # Compress the data
        transforms = ComposeTransforms([ApplySVD(basis, inverse=False)])
        compressed = apply_transforms_to_polarizations(
            sample_polarizations, transforms
        )

        dataset = NewWaveformDataset(
            parameters=parameters,
            polarizations=compressed,
            svd_basis=basis,
        )

        path = tmp_path / "compressed_dataset.hdf5"
        dataset.save(path)
        loaded = NewWaveformDataset.load(path)

        assert loaded.svd_basis is not None
        assert loaded.svd_basis.n_components == 10
        assert np.allclose(loaded.svd_basis.V, basis.V)
        assert np.allclose(loaded.polarizations.h_plus, compressed.h_plus)
