"""Tests for new-style dataset generation.

Uses RandomApproximant since LAL-based approximants are not fully ported
in the new API.
"""

import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator.new_api import build_waveform_generator
from dingo.gw.waveform_generator.polarizations import BatchPolarizations
from dingo.gw.dataset.new_generate import (
    generate_waveforms_sequential,
    new_generate_parameters_and_polarizations,
    apply_transforms_to_polarizations,
    train_svd_basis,
)


@pytest.fixture
def domain():
    return UniformFrequencyDomain(f_min=20.0, f_max=256.0, delta_f=1.0)


@pytest.fixture
def waveform_generator(domain):
    config = {
        "approximant": "RandomApproximant",
        "f_ref": 20.0,
    }
    return build_waveform_generator(config, domain)


@pytest.fixture
def random_bbh_parameters():
    """DataFrame with BBH parameters compatible with RandomApproximant."""
    np.random.seed(42)
    n = 5
    return pd.DataFrame(
        {
            "mass_1": np.random.uniform(20, 50, n),
            "mass_2": np.random.uniform(10, 30, n),
            "luminosity_distance": np.random.uniform(100, 1000, n),
            "phase": np.random.uniform(0, 2 * np.pi, n),
        }
    )


class TestGenerateWaveformsSequential:

    def test_basic_generation(self, waveform_generator, random_bbh_parameters):
        polarizations = generate_waveforms_sequential(
            waveform_generator, random_bbh_parameters
        )

        assert isinstance(polarizations, BatchPolarizations)
        assert polarizations.h_plus.shape[0] == 5
        assert polarizations.h_cross.shape[0] == 5
        assert not np.any(np.isnan(polarizations.h_plus))

    def test_empty_parameters(self, waveform_generator):
        parameters = pd.DataFrame()
        polarizations = generate_waveforms_sequential(
            waveform_generator, parameters
        )
        assert len(polarizations) == 0


class TestGenerateParametersAndPolarizations:

    def test_basic(self, waveform_generator):
        from bilby.core.prior import Uniform

        # Use a simple prior that samples mass_1, mass_2, luminosity_distance, phase
        from bilby.gw.prior import BBHPriorDict

        prior = BBHPriorDict(
            {
                "mass_1": Uniform(minimum=20, maximum=50, name="mass_1"),
                "mass_2": Uniform(minimum=10, maximum=30, name="mass_2"),
                "luminosity_distance": Uniform(
                    minimum=100, maximum=1000, name="luminosity_distance"
                ),
                "phase": Uniform(minimum=0, maximum=6.28, name="phase"),
            }
        )

        parameters, polarizations = new_generate_parameters_and_polarizations(
            waveform_generator, prior, num_samples=5
        )

        assert isinstance(parameters, pd.DataFrame)
        assert isinstance(polarizations, BatchPolarizations)
        assert len(parameters) == 5
        assert len(polarizations) == 5


class TestApplyTransforms:

    def test_none_transform(self):
        np.random.seed(42)
        polarizations = BatchPolarizations(
            h_plus=np.random.randn(5, 100) + 1j * np.random.randn(5, 100),
            h_cross=np.random.randn(5, 100) + 1j * np.random.randn(5, 100),
        )
        result = apply_transforms_to_polarizations(polarizations, None)
        assert np.array_equal(result.h_plus, polarizations.h_plus)

    def test_with_svd_transform(self):
        from dingo.gw.compression.svd import SVDBasis
        from dingo.gw.compression.transforms import ApplySVD, ComposeTransforms

        np.random.seed(42)
        n_freq = 100
        data = np.random.randn(50, n_freq) + 1j * np.random.randn(50, n_freq)
        basis = SVDBasis()
        basis.generate_basis(data, n_components=10, method="scipy")

        transforms = ComposeTransforms([ApplySVD(basis, inverse=False)])

        polarizations = BatchPolarizations(
            h_plus=np.random.randn(5, n_freq) + 1j * np.random.randn(5, n_freq),
            h_cross=np.random.randn(5, n_freq) + 1j * np.random.randn(5, n_freq),
        )

        result = apply_transforms_to_polarizations(polarizations, transforms)
        assert result.h_plus.shape == (5, 10)
        assert result.h_cross.shape == (5, 10)


class TestTrainSVDBasis:

    def test_basic(self):
        np.random.seed(42)
        n_waveforms = 30
        n_freq = 100
        polarizations = BatchPolarizations(
            h_plus=np.random.randn(n_waveforms, n_freq)
            + 1j * np.random.randn(n_waveforms, n_freq),
            h_cross=np.random.randn(n_waveforms, n_freq)
            + 1j * np.random.randn(n_waveforms, n_freq),
        )
        parameters = pd.DataFrame(
            {"mass_1": np.random.uniform(10, 50, n_waveforms)}
        )

        basis, n_train, n_val = train_svd_basis(
            polarizations, parameters, size=10, n_train=20
        )

        assert basis.n_components == 10
        assert n_train == 20
        assert n_val == 10
