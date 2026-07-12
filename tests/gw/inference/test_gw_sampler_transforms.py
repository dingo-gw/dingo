"""Tests for GWSampler._initialize_transforms with and without tokenization."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from dingo.core.transforms import GetItem
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.transforms import (
    StrainTokenization,
    SelectKeys,
    ToTorch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DETECTORS = ["H1", "L1"]
STANDARDIZATION = {
    "mean": {"chirp_mass": 20.0},
    "std": {"chirp_mass": 5.0},
}
INFERENCE_PARAMS = ["chirp_mass"]


def _make_domain(f_min=20.0, f_max=128.0, delta_f=0.25):
    return UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=delta_f)


def _make_sampler_stub(domain, tokenization_settings=None):
    """Return a GWSampler with the minimum attributes set to call _initialize_transforms.

    Uses object.__setattr__ to bypass Sampler.__init__, so no real model or dataset
    is needed.
    """
    data_settings = {
        "detectors": DETECTORS,
        "standardization": STANDARDIZATION,
        "ref_time": 1126259462.391,
    }
    if tokenization_settings is not None:
        data_settings["tokenization"] = tokenization_settings

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    sampler = object.__new__(GWSampler)
    sampler.domain = domain
    sampler.model = mock_model
    metadata = {
        "train_settings": {"data": data_settings},
        "dataset_settings": {"intrinsic_prior": {}},
    }
    sampler.metadata = metadata
    # GWSamplerMixin.detectors reads from base_model_metadata (== metadata for non-GNPE).
    sampler.base_model_metadata = metadata
    sampler.inference_parameters = INFERENCE_PARAMS
    sampler._minimum_frequency = None
    sampler._maximum_frequency = None
    sampler._suppress = None
    return sampler


def _make_context(domain, rng=None):
    """Build a minimal {'waveform': ..., 'asds': ...} dict for *domain*."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(domain.sample_frequencies)
    return {
        "waveform": {
            d: (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
                np.complex64
            )
            for d in DETECTORS
        },
        "asds": {d: rng.uniform(1e-24, 1e-23, n).astype(np.float32) for d in DETECTORS},
    }


# ---------------------------------------------------------------------------
# _initialize_transforms — resnet (no tokenization)
# ---------------------------------------------------------------------------


def test_resnet_path_uses_get_item():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=None)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    assert isinstance(transforms[-1], GetItem)
    assert transforms[-1].key == "waveform"


def test_resnet_path_has_no_strain_tokenization():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=None)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert StrainTokenization not in types
    assert SelectKeys not in types


def test_resnet_path_output_is_tensor():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain)
    sampler._initialize_transforms()

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    assert isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# _initialize_transforms — transformer (with tokenization)
# ---------------------------------------------------------------------------

TOK_SETTINGS = {
    "token_size": 16,
    "num_tokens_per_block": None,
    "drop_last_token": False,
}


def test_transformer_path_has_strain_tokenization():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert StrainTokenization in types


def test_transformer_path_has_select_keys():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert SelectKeys in types


def test_transformer_path_no_get_item():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert GetItem not in types


def test_transformer_tokenization_precedes_to_torch():
    """StrainTokenization must come before ToTorch in the chain."""
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    indices = {type(t): i for i, t in enumerate(transforms)}
    assert indices[StrainTokenization] < indices[ToTorch]


def test_transformer_select_keys_follows_to_torch():
    """SelectKeys must come after ToTorch in the chain."""
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    indices = {type(t): i for i, t in enumerate(transforms)}
    assert indices[SelectKeys] > indices[ToTorch]


def test_transformer_path_output_is_dict_of_three_tensors():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    assert isinstance(x, dict)
    assert list(x) == ["waveform", "position", "drop_token_mask"]
    assert all(isinstance(v, torch.Tensor) for v in x.values())
    assert x["drop_token_mask"].dtype == torch.bool


def test_transformer_path_waveform_and_position_num_tokens_match():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    assert (
        x["waveform"].shape[0]
        == x["position"].shape[0]
        == x["drop_token_mask"].shape[0]
    )


# ---------------------------------------------------------------------------
# Token suppression (inference-side frequency updates for tokenized models)
# ---------------------------------------------------------------------------

TOK_WITH_DROP = {
    **TOK_SETTINGS,
    "drop_frequency_range": {"f_cut": {"p_cut": 0.25}},
}


def test_suppress_requires_tokenized_model_with_drop_augmentation():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=None)
    with pytest.raises(ValueError, match="tokenized"):
        sampler.suppress = [50.0, 60.0]

    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    with pytest.raises(ValueError, match="drop augmentation"):
        sampler.suppress = [50.0, 60.0]


def test_suppress_validates_interval():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_WITH_DROP)
    with pytest.raises(ValueError, match="f_lo < f_hi"):
        sampler.suppress = [60.0, 50.0]
    with pytest.raises(ValueError, match="f_lo < f_hi"):
        sampler.suppress = [5.0, 60.0]  # below domain f_min
    with pytest.raises(ValueError, match="Unknown detectors"):
        sampler.suppress = {"V1": [50.0, 60.0]}


def test_suppress_masks_tokens():
    from dingo.gw.transforms import UpdateFrequencyRange

    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_WITH_DROP)
    sampler.suppress = [50.0, 60.0]
    assert sampler.frequency_updates

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert UpdateFrequencyRange in types

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    mask = x["drop_token_mask"].numpy()
    position = x["position"].numpy()
    overlaps = (position[..., 1] >= 50.0) & (position[..., 0] <= 60.0)
    assert np.array_equal(mask, overlaps)
    assert mask.any() and not mask.all()


def test_no_frequency_update_has_no_update_transform():
    from dingo.gw.transforms import UpdateFrequencyRange

    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_WITH_DROP)
    sampler._initialize_transforms()
    types = [type(t) for t in sampler.transform_pre.transforms]
    assert UpdateFrequencyRange not in types
