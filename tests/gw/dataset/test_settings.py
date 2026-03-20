"""Tests for new-style dataset settings dataclasses.

Ported from dingo-waveform tests/test_dataset_generation.py (settings portion).
"""

import pytest

from dingo.gw.dataset.compression_settings import CompressionSettings, SVDSettings
from dingo.gw.dataset.generation_types import WaveformGeneratorConfig, WaveformResult
from dingo.gw.dataset.waveform_generator_settings import WaveformGeneratorSettings
from dingo.gw.dataset.dataset_settings import DatasetSettings

import numpy as np


class TestWaveformGeneratorSettings:

    def test_basic_creation(self):
        settings = WaveformGeneratorSettings(
            approximant="IMRPhenomD", f_ref=20.0
        )
        assert str(settings.approximant) == "IMRPhenomD"
        assert settings.f_ref == 20.0
        assert settings.spin_conversion_phase is None
        assert settings.f_start is None

    def test_with_optional_fields(self):
        settings = WaveformGeneratorSettings(
            approximant="IMRPhenomXPHM",
            f_ref=20.0,
            spin_conversion_phase=0.5,
            f_start=10.0,
        )
        assert settings.spin_conversion_phase == 0.5
        assert settings.f_start == 10.0

    def test_invalid_f_ref(self):
        with pytest.raises(ValueError, match="f_ref must be positive"):
            WaveformGeneratorSettings(approximant="IMRPhenomD", f_ref=-1.0)

    def test_invalid_f_start(self):
        with pytest.raises(ValueError, match="f_start must be positive"):
            WaveformGeneratorSettings(
                approximant="IMRPhenomD", f_ref=20.0, f_start=-5.0
            )

    def test_to_dict(self):
        settings = WaveformGeneratorSettings(
            approximant="IMRPhenomD",
            f_ref=20.0,
            spin_conversion_phase=0.0,
        )
        d = settings.to_dict()
        assert d == {
            "approximant": "IMRPhenomD",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        }

    def test_to_dict_omits_none(self):
        settings = WaveformGeneratorSettings(
            approximant="IMRPhenomD", f_ref=20.0
        )
        d = settings.to_dict()
        assert "spin_conversion_phase" not in d
        assert "f_start" not in d


class TestSVDSettings:

    def test_basic_creation(self):
        settings = SVDSettings(size=50, num_training_samples=1000)
        assert settings.size == 50
        assert settings.num_training_samples == 1000
        assert settings.num_validation_samples == 0
        assert settings.file is None

    def test_negative_size(self):
        with pytest.raises(ValueError, match="non-negative"):
            SVDSettings(size=-1, num_training_samples=100)

    def test_zero_training_samples(self):
        with pytest.raises(ValueError, match="positive"):
            SVDSettings(size=50, num_training_samples=0)

    def test_negative_validation_samples(self):
        with pytest.raises(ValueError, match="non-negative"):
            SVDSettings(size=50, num_training_samples=100, num_validation_samples=-1)

    def test_file_string_to_path(self):
        settings = SVDSettings(
            size=50, num_training_samples=100, file="/tmp/svd.hdf5"
        )
        from pathlib import Path

        assert isinstance(settings.file, Path)


class TestCompressionSettings:

    def test_svd_only(self):
        svd = SVDSettings(size=50, num_training_samples=1000)
        settings = CompressionSettings(svd=svd)
        assert settings.svd is svd
        assert settings.whitening is None

    def test_whitening_only(self):
        settings = CompressionSettings(whitening="/tmp/asd.txt")
        from pathlib import Path

        assert isinstance(settings.whitening, Path)
        assert settings.svd is None

    def test_both(self):
        svd = SVDSettings(size=50, num_training_samples=1000)
        settings = CompressionSettings(svd=svd, whitening="/tmp/asd.txt")
        assert settings.svd is svd
        assert settings.whitening is not None

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CompressionSettings()


class TestWaveformGeneratorConfig:

    def test_creation(self):
        config = WaveformGeneratorConfig(
            approximant="IMRPhenomD", f_ref=20.0, spin_conversion_phase=0.0
        )
        assert config.approximant == "IMRPhenomD"

    def test_to_dict(self):
        config = WaveformGeneratorConfig(
            approximant="IMRPhenomD", f_ref=20.0, spin_conversion_phase=0.0
        )
        d = config.to_dict()
        assert d == {
            "approximant": "IMRPhenomD",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        }


class TestWaveformResult:

    def test_success(self):
        hp = np.array([1.0, 2.0])
        hc = np.array([3.0, 4.0])
        result = WaveformResult.success_result(hp, hc)
        assert result.success
        assert result.error_message is None
        assert np.array_equal(result.h_plus, hp)

    def test_failure(self):
        result = WaveformResult.failure_result("test error")
        assert not result.success
        assert result.h_plus is None
        assert result.h_cross is None
        assert result.error_message == "test error"

    def test_to_dict(self):
        hp = np.array([1.0])
        hc = np.array([2.0])
        result = WaveformResult.success_result(hp, hc)
        d = result.to_dict()
        assert "h_plus" in d
        assert "h_cross" in d


class TestDatasetSettings:

    @pytest.fixture
    def domain_config(self):
        return {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 512.0,
            "delta_f": 0.125,
        }

    @pytest.fixture
    def wfg_config(self):
        return {
            "approximant": "IMRPhenomD",
            "f_ref": 20.0,
        }

    @pytest.fixture
    def prior_config(self):
        return {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=50.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=50.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.2, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary='periodic')",
            "chi_1": "bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.88))",
            "chi_2": "bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.88))",
            "geocent_time": 0.0,
        }

    def test_from_dict(self, domain_config, wfg_config, prior_config):
        settings = DatasetSettings.from_dict(
            {
                "domain": domain_config,
                "waveform_generator": wfg_config,
                "intrinsic_prior": prior_config,
                "num_samples": 5,
            }
        )
        assert settings.num_samples == 5
        from dingo.gw.domains import DomainParameters

        assert isinstance(settings.domain, DomainParameters)

    def test_invalid_num_samples(self, domain_config, wfg_config, prior_config):
        with pytest.raises(ValueError, match="num_samples.*positive"):
            DatasetSettings.from_dict(
                {
                    "domain": domain_config,
                    "waveform_generator": wfg_config,
                    "intrinsic_prior": prior_config,
                    "num_samples": 0,
                }
            )

    def test_to_dict(self, domain_config, wfg_config, prior_config):
        settings = DatasetSettings.from_dict(
            {
                "domain": domain_config,
                "waveform_generator": wfg_config,
                "intrinsic_prior": prior_config,
                "num_samples": 10,
            }
        )
        d = settings.to_dict()
        assert isinstance(d, dict)
        assert d["num_samples"] == 10
        assert "domain" in d
        assert "waveform_generator" in d

    def test_validate_is_noop(self, domain_config, wfg_config, prior_config):
        settings = DatasetSettings.from_dict(
            {
                "domain": domain_config,
                "waveform_generator": wfg_config,
                "intrinsic_prior": prior_config,
                "num_samples": 5,
            }
        )
        settings.validate()  # Should not raise
