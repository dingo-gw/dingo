"""
Tests for the validation guard around data.time_alignment=True. The validation
is decoupled from the rest of set_train_transforms so it can be exercised
without building a WaveformDataset.
"""

import pytest

from dingo.gw.training.train_builders import validate_time_alignment_settings


def _base_settings():
    return {
        "inference_parameters": ["mass_1", "mass_2", "luminosity_distance"],
        "conditioning_parameters": ["ra", "dec", "geocent_time"],
    }


def test_validate_time_alignment_accepts_well_formed_settings():
    validate_time_alignment_settings(_base_settings())


@pytest.mark.parametrize("missing", ["ra", "dec", "geocent_time"])
def test_validate_time_alignment_rejects_missing_conditioning(missing):
    s = _base_settings()
    s["conditioning_parameters"] = [
        p for p in s["conditioning_parameters"] if p != missing
    ]
    with pytest.raises(ValueError, match=missing):
        validate_time_alignment_settings(s)


@pytest.mark.parametrize("inf_param", ["ra", "dec", "geocent_time"])
def test_validate_time_alignment_rejects_conditioning_in_inference(inf_param):
    s = _base_settings()
    s["inference_parameters"] = s["inference_parameters"] + [inf_param]
    with pytest.raises(ValueError, match=inf_param):
        validate_time_alignment_settings(s)


def test_validate_time_alignment_rejects_gnpe_combination():
    s = _base_settings()
    s["gnpe_time_shifts"] = {
        "kernel": "bilby.core.prior.Uniform(0, 1)",
        "exact_equiv": True,
    }
    with pytest.raises(ValueError, match="gnpe_time_shifts"):
        validate_time_alignment_settings(s)


def test_validate_time_alignment_handles_missing_conditioning_key():
    """If conditioning_parameters is absent entirely (e.g., user forgot to add it),
    we should still get a clear error listing all three required names."""
    s = {"inference_parameters": ["mass_1"]}
    with pytest.raises(ValueError, match="ra"):
        validate_time_alignment_settings(s)
