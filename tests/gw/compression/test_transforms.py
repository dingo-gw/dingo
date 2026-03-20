"""Tests for compression transforms.

Ported from dingo-waveform tests/test_transforms.py.
"""

import h5py
import numpy as np
import pytest

from dingo.gw.compression.svd import SVDBasis
from dingo.gw.compression.transforms import (
    ApplySVD,
    ComposeTransforms,
    Transform,
    WhitenAndUnwhiten,
)
from dingo.gw.domains import UniformFrequencyDomain


class TestApplySVD:

    @pytest.fixture
    def sample_basis(self):
        np.random.seed(42)
        data = np.random.randn(100, 200) + 1j * np.random.randn(100, 200)
        basis = SVDBasis()
        basis.generate_basis(data, n_components=20, method="scipy")
        return basis

    @pytest.fixture
    def sample_polarizations(self):
        np.random.seed(42)
        return {
            "h_plus": np.random.randn(10, 200) + 1j * np.random.randn(10, 200),
            "h_cross": np.random.randn(10, 200)
            + 1j * np.random.randn(10, 200),
        }

    def test_compress_transform(self, sample_basis, sample_polarizations):
        transform = ApplySVD(sample_basis, inverse=False)
        result = transform(sample_polarizations)

        assert "h_plus" in result
        assert "h_cross" in result
        assert result["h_plus"].shape == (10, 20)
        assert result["h_cross"].shape == (10, 20)

    def test_decompress_transform(self, sample_basis):
        compressed = {
            "h_plus": np.random.randn(10, 20) + 1j * np.random.randn(10, 20),
            "h_cross": np.random.randn(10, 20)
            + 1j * np.random.randn(10, 20),
        }
        transform = ApplySVD(sample_basis, inverse=True)
        result = transform(compressed)

        assert result["h_plus"].shape == (10, 200)
        assert result["h_cross"].shape == (10, 200)

    def test_roundtrip(self, sample_basis, sample_polarizations):
        compress = ApplySVD(sample_basis, inverse=False)
        decompress = ApplySVD(sample_basis, inverse=True)

        compressed = compress(sample_polarizations)
        reconstructed = decompress(compressed)

        assert (
            reconstructed["h_plus"].shape
            == sample_polarizations["h_plus"].shape
        )
        assert reconstructed["h_plus"].dtype == sample_polarizations["h_plus"].dtype
        assert not np.any(np.isnan(reconstructed["h_plus"]))

    def test_error_untrained_basis(self):
        basis = SVDBasis()
        with pytest.raises(ValueError, match="has not been trained"):
            ApplySVD(basis, inverse=False)

    def test_repr(self, sample_basis):
        transform = ApplySVD(sample_basis, inverse=False)
        repr_str = repr(transform)
        assert "ApplySVD" in repr_str
        assert "n_components=20" in repr_str
        assert "compress" in repr_str


class TestWhitenAndUnwhiten:

    @pytest.fixture
    def asd_file(self, tmp_path):
        domain = UniformFrequencyDomain(
            f_min=20.0, f_max=1024.0, delta_f=0.25
        )
        asd = np.ones(len(domain)) * 1e-23

        asd_path = tmp_path / "test_asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=asd)

        return domain, asd_path

    @pytest.fixture
    def sample_polarizations_whitening(self, asd_file):
        domain, _ = asd_file
        np.random.seed(42)
        return {
            "h_plus": np.random.randn(5, len(domain))
            + 1j * np.random.randn(5, len(domain)),
            "h_cross": np.random.randn(5, len(domain))
            + 1j * np.random.randn(5, len(domain)),
        }

    def test_whiten_transform(self, asd_file, sample_polarizations_whitening):
        domain, asd_path = asd_file
        transform = WhitenAndUnwhiten(domain, asd_path, inverse=False)
        result = transform(sample_polarizations_whitening)

        assert "h_plus" in result
        assert "h_cross" in result
        assert (
            result["h_plus"].shape
            == sample_polarizations_whitening["h_plus"].shape
        )

    def test_unwhiten_transform(
        self, asd_file, sample_polarizations_whitening
    ):
        domain, asd_path = asd_file
        transform = WhitenAndUnwhiten(domain, asd_path, inverse=True)
        result = transform(sample_polarizations_whitening)
        assert (
            result["h_plus"].shape
            == sample_polarizations_whitening["h_plus"].shape
        )

    def test_whitening_roundtrip(
        self, asd_file, sample_polarizations_whitening
    ):
        domain, asd_path = asd_file
        whiten = WhitenAndUnwhiten(domain, asd_path, inverse=False)
        unwhiten = WhitenAndUnwhiten(domain, asd_path, inverse=True)

        whitened = whiten(sample_polarizations_whitening)
        reconstructed = unwhiten(whitened)

        assert np.allclose(
            reconstructed["h_plus"],
            sample_polarizations_whitening["h_plus"],
        )
        assert np.allclose(
            reconstructed["h_cross"],
            sample_polarizations_whitening["h_cross"],
        )

    def test_error_nonexistent_file(self):
        domain = UniformFrequencyDomain(
            f_min=20.0, f_max=1024.0, delta_f=0.25
        )
        with pytest.raises(FileNotFoundError):
            WhitenAndUnwhiten(domain, "/nonexistent/asd.hdf5", inverse=False)

    def test_error_wrong_asd_length(self, tmp_path):
        domain = UniformFrequencyDomain(
            f_min=20.0, f_max=1024.0, delta_f=0.25
        )
        asd_path = tmp_path / "wrong_asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=np.ones(100))

        with pytest.raises(ValueError, match="does not match domain length"):
            WhitenAndUnwhiten(domain, asd_path, inverse=False)

    def test_repr(self, asd_file):
        domain, asd_path = asd_file
        transform = WhitenAndUnwhiten(domain, asd_path, inverse=False)
        repr_str = repr(transform)
        assert "WhitenAndUnwhiten" in repr_str
        assert "whiten" in repr_str


class TestComposeTransforms:

    @pytest.fixture
    def mock_transform(self):
        class MockTransform(Transform):
            def __init__(self, multiplier):
                self.multiplier = multiplier

            def __call__(self, polarizations):
                return {
                    key: value * self.multiplier
                    for key, value in polarizations.items()
                }

        return MockTransform

    def test_empty_composition(self):
        compose = ComposeTransforms([])
        data = {
            "h_plus": np.array([1, 2, 3]),
            "h_cross": np.array([4, 5, 6]),
        }
        result = compose(data)
        assert np.array_equal(result["h_plus"], data["h_plus"])
        assert np.array_equal(result["h_cross"], data["h_cross"])

    def test_single_transform(self, mock_transform):
        compose = ComposeTransforms([mock_transform(2.0)])
        data = {
            "h_plus": np.array([1, 2, 3]),
            "h_cross": np.array([4, 5, 6]),
        }
        result = compose(data)
        assert np.array_equal(result["h_plus"], np.array([2, 4, 6]))
        assert np.array_equal(result["h_cross"], np.array([8, 10, 12]))

    def test_multiple_transforms(self, mock_transform):
        compose = ComposeTransforms(
            [mock_transform(2.0), mock_transform(3.0)]
        )
        data = {"h_plus": np.array([1.0]), "h_cross": np.array([2.0])}
        result = compose(data)
        assert np.allclose(result["h_plus"], np.array([6.0]))
        assert np.allclose(result["h_cross"], np.array([12.0]))

    def test_realistic_pipeline(self, tmp_path):
        domain = UniformFrequencyDomain(
            f_min=20.0, f_max=512.0, delta_f=0.5
        )
        asd = np.ones(len(domain)) * 1e-23

        asd_path = tmp_path / "asd.hdf5"
        with h5py.File(asd_path, "w") as f:
            f.create_dataset("asd", data=asd)

        np.random.seed(42)
        train_data = np.random.randn(50, len(domain)) + 1j * np.random.randn(
            50, len(domain)
        )
        basis = SVDBasis()
        basis.generate_basis(train_data, n_components=20, method="scipy")

        pipeline = ComposeTransforms(
            [
                WhitenAndUnwhiten(domain, asd_path, inverse=False),
                ApplySVD(basis, inverse=False),
            ]
        )

        data = {
            "h_plus": np.random.randn(5, len(domain))
            + 1j * np.random.randn(5, len(domain)),
            "h_cross": np.random.randn(5, len(domain))
            + 1j * np.random.randn(5, len(domain)),
        }

        result = pipeline(data)
        assert result["h_plus"].shape == (5, 20)
        assert result["h_cross"].shape == (5, 20)

    def test_repr(self, mock_transform):
        compose = ComposeTransforms(
            [mock_transform(2.0), mock_transform(3.0)]
        )
        repr_str = repr(compose)
        assert "ComposeTransforms" in repr_str
        assert "MockTransform" in repr_str
