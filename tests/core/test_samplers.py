import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch
from torchvision.transforms import Compose

from dingo.core.samplers import FixedInitSampler


class TestFixedInitSampler:
    """Tests for the FixedInitSampler class."""

    def test_basic_construction(self):
        params = {"chirp_mass_proxy": 1.2, "foo": 3.0}
        sampler = FixedInitSampler(init_parameters=params)
        assert sampler.init_parameters == params
        assert sampler._log_prob == 0.0
        assert sampler.unconditional_model is False
        assert sampler.metadata == {}
        assert isinstance(sampler.transform_pre, Compose)

    def test_custom_log_prob(self):
        sampler = FixedInitSampler(
            init_parameters={"x": 1.0}, log_prob=-5.0
        )
        assert sampler._log_prob == -5.0

    def test_run_sampler_returns_correct_shape(self):
        params = {"chirp_mass_proxy": 1.2, "mass_ratio_proxy": 0.8}
        sampler = FixedInitSampler(init_parameters=params)
        num_samples = 50
        samples = sampler._run_sampler(num_samples)

        assert set(samples.keys()) == {"chirp_mass_proxy", "mass_ratio_proxy", "log_prob"}
        for key in samples:
            assert samples[key].shape == (num_samples,)

    def test_run_sampler_returns_correct_values(self):
        params = {"chirp_mass_proxy": 1.2, "mass_ratio_proxy": 0.8}
        sampler = FixedInitSampler(init_parameters=params, log_prob=-3.0)
        samples = sampler._run_sampler(10)

        assert torch.allclose(samples["chirp_mass_proxy"], torch.ones(10) * 1.2)
        assert torch.allclose(samples["mass_ratio_proxy"], torch.ones(10) * 0.8)
        assert torch.allclose(samples["log_prob"], torch.ones(10) * (-3.0))

    def test_run_sampler_single_sample(self):
        params = {"x": 42.0}
        sampler = FixedInitSampler(init_parameters=params)
        samples = sampler._run_sampler(1)
        assert samples["x"].shape == (1,)
        assert samples["x"].item() == 42.0

    def test_run_sampler_ignores_extra_args(self):
        """_run_sampler should accept and ignore extra positional/keyword args
        (matching the Sampler._run_sampler signature which receives context)."""
        params = {"x": 1.0}
        sampler = FixedInitSampler(init_parameters=params)
        samples = sampler._run_sampler(5, "ignored_context", extra_kwarg=True)
        assert samples["x"].shape == (5,)

    def test_empty_parameters(self):
        """FixedInitSampler with no parameters should still return log_prob."""
        sampler = FixedInitSampler(init_parameters={})
        samples = sampler._run_sampler(3)
        assert set(samples.keys()) == {"log_prob"}
        assert samples["log_prob"].shape == (3,)
