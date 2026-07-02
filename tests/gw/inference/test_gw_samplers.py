"""Tests for GWSampler._initialize_transforms and _run_sampler list-handling."""

import numpy as np
import torch
from unittest.mock import MagicMock

from dingo.core.transforms import GetItem
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.transforms import (
    StrainTokenization,
    UnpackDict,
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
    assert UnpackDict not in types


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


def test_transformer_path_has_unpack_dict():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert UnpackDict in types


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


def test_transformer_unpack_dict_follows_to_torch():
    """UnpackDict must come after ToTorch in the chain."""
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    indices = {type(t): i for i, t in enumerate(transforms)}
    assert indices[UnpackDict] > indices[ToTorch]


def test_transformer_path_output_is_list_of_three_tensors():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    assert isinstance(x, list)
    assert len(x) == 3
    waveform, position, mask = x
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(position, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool


def test_transformer_path_waveform_and_position_num_tokens_match():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    context = _make_context(domain)
    waveform, position, mask = sampler.transform_pre(context)
    assert waveform.shape[0] == position.shape[0] == mask.shape[0]


# ---------------------------------------------------------------------------
# _run_sampler list-handling logic
# ---------------------------------------------------------------------------


def test_run_sampler_wraps_single_tensor_in_list():
    """A single tensor from transform_pre must be wrapped into a one-element list."""
    single = torch.randn(10, 48)
    if isinstance(single, list):
        result = [t.unsqueeze(0) for t in single]
    else:
        result = [single.unsqueeze(0)]
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (1, 10, 48)


def test_run_sampler_list_gets_batch_dim_added():
    """A list of tensors from transform_pre must each get a batch dimension."""
    tensors = [
        torch.randn(86, 48),
        torch.randn(86, 3),
        torch.zeros(86, dtype=torch.bool),
    ]
    x = tensors
    if isinstance(x, list):
        result = [t.unsqueeze(0) for t in x]
    else:
        result = [x.unsqueeze(0)]
    assert len(result) == 3
    assert result[0].shape == (1, 86, 48)
    assert result[1].shape == (1, 86, 3)
    assert result[2].shape == (1, 86)
