from pathlib import Path
from typing import get_args

import pytest
import torch

from dingo.core.utils.backward_compatibility import (
    Device,
    torch_available_devices,
    torch_load_with_fallback,
    update_data_config,
    update_model_config,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)


@pytest.fixture
def model_path(tmp_path) -> str:
    model = DummyModel()
    model.linear.weight.data.fill_(1.0)
    model.linear.bias.data.fill_(2.0)
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)
    return str(model_path)


def test_torch_available_device() -> None:
    """
    checks torch_available_devices do not
    raise an error and lists at least cpu
    """
    devices = torch_available_devices()
    assert len(devices) != 0
    assert "cpu" in devices


def test_load_torch_with_fallback(model_path) -> None:
    """
    checks torch_load_with_fallback do not
    raise an error and map at least correctly to cpu
    """
    devices = get_args(Device)
    model, device = torch_load_with_fallback(model_path, preferred_map_location="cpu")
    assert device == torch.device("cpu")
    for device in devices:
        # just checking no error is raised.
        # Not checking if it is mapped to the proper device, as we do not
        # want to assume which device is available on the test machine.
        torch_load_with_fallback(model_path, preferred_map_location=device)


# ---------------------------------------------------------------------------
# update_model_config — embedding_type backfill
# ---------------------------------------------------------------------------


def test_update_model_config_backfills_embedding_type_resnet():
    """Old checkpoints with embedding_kwargs but no embedding_type get resnet injected."""
    settings = {"posterior_model_type": "normalizing_flow", "embedding_kwargs": {}}
    update_model_config(settings)
    assert settings["embedding_type"] == "resnet"


def test_update_model_config_preserves_existing_embedding_type():
    """An explicit embedding_type (e.g. transformer) must not be overwritten."""
    settings = {
        "posterior_model_type": "normalizing_flow",
        "embedding_kwargs": {},
        "embedding_type": "transformer",
    }
    update_model_config(settings)
    assert settings["embedding_type"] == "transformer"


def test_update_model_config_no_embedding_type_without_embedding_kwargs():
    """Settings without embedding_kwargs must not get an embedding_type injected."""
    settings = {"posterior_model_type": "normalizing_flow"}
    update_model_config(settings)
    assert "embedding_type" not in settings


# ---------------------------------------------------------------------------
# update_model_config — dingo-t1 transformer kwargs
# ---------------------------------------------------------------------------


def _dingo_t1_model_settings():
    """Model settings as stored by the published Dingo-T1 network (dingo-t1 branch)."""
    return {
        "posterior_model_type": "normalizing_flow",
        "embedding_type": "transformer",
        "embedding_kwargs": {
            "tokenizer_kwargs": {
                "condition_on_position": True,
                "context_in_initial_layer": False,
                "hidden_dims": [512],
                "activation": "elu",
                "batch_norm": False,
                "layer_norm": True,
                "input_dims": [207, 48],
                "output_dim": 1024,
                "context_features": 5,
                "num_blocks": 3,
            },
            "transformer_kwargs": {"d_model": 1024, "num_layers": 8},
            "pooling": "cls",
            "final_net_kwargs": {
                "activation": "elu",
                "output_dim": 128,
                "input_dim": 1024,
            },
            "added_context": False,
        },
    }


def test_update_model_config_maps_dingo_t1_transformer_kwargs():
    """Fixed/derived dingo-t1 kwargs are dropped; everything else is kept."""
    settings = _dingo_t1_model_settings()
    update_model_config(settings)
    embedding_kwargs = settings["embedding_kwargs"]
    assert "added_context" not in embedding_kwargs
    assert embedding_kwargs["tokenizer_kwargs"] == {
        "hidden_dims": [512],
        "activation": "elu",
        "batch_norm": False,
        "layer_norm": True,
        "input_dim": 48,
        "num_blocks": 3,
    }
    assert embedding_kwargs["final_net_kwargs"] == {
        "activation": "elu",
        "output_dim": 128,
    }
    update_model_config(settings)  # idempotent
    assert embedding_kwargs["tokenizer_kwargs"]["num_blocks"] == 3


def test_update_model_config_rejects_unsupported_dingo_t1_tokenizer():
    """Tokenizer variants that no longer exist must fail loudly, not load silently."""
    settings = _dingo_t1_model_settings()
    settings["embedding_kwargs"]["tokenizer_kwargs"]["context_in_initial_layer"] = True
    with pytest.raises(ValueError, match="context_in_initial_layer"):
        update_model_config(settings)


# ---------------------------------------------------------------------------
# update_data_config — dingo-t1 tokenization settings
# ---------------------------------------------------------------------------


def dingo_t1_settings():
    """Metadata with the data settings stored by the published Dingo-T1 network
    (dingo-t1 branch)."""
    return {
        "train_settings": {
            "data": {
                "detectors": ["H1", "L1", "V1"],
                "tokenization": {
                    "token_size": 16,
                    "drop_detectors": {
                        "p_drop_012_detectors": [0.6, 0.3, 0.1],
                        "p_drop_hlv": {"H1": 0.3, "L1": 0.3, "V1": 0.4},
                    },
                    "drop_frequency_range": {
                        "f_cut": {
                            "p_cut": 0.25,
                            "f_max_lower_cut": 180.0,
                            "f_min_upper_cut": 80.0,
                            "p_same_cut_all_detectors": 0.7,
                            "p_lower_upper_both": [0.1, 0.7, 0.2],
                        },
                        "mask_interval": {
                            "p_per_detector": 0.1,
                            "f_min": 20.0,
                            "f_max": 1800.0,
                            "max_width": 10.0,
                        },
                    },
                },
            }
        }
    }


DINGO_T1_TOKENIZATION_CONVERTED = {
    "token_size": 16,
    "mask_detectors": {
        "num_blocks": 3,
        "p_mask_012_detectors": [0.6, 0.3, 0.1],
        "p_mask_hlv": {"H1": 0.3, "L1": 0.3, "V1": 0.4},
    },
    "mask_frequency_range": {
        "p_mask": 0.25,
        "p_same_all_detectors": 0.7,
        "p_lower_upper_both": [0.1, 0.7, 0.2],
        "f_min_upper": 180.0,
        "f_max_lower": 80.0,
    },
    "mask_frequency_notches": {
        "p_per_detector": 0.1,
        "f_min": 20.0,
        "f_max": 1800.0,
        "max_width": 10.0,
    },
}


def test_update_data_config_maps_dingo_t1_schema():
    """Old keys are renamed in place and the conversion is idempotent."""
    settings = dingo_t1_settings()
    update_data_config(settings)
    assert (
        settings["train_settings"]["data"]["tokenization"]
        == DINGO_T1_TOKENIZATION_CONVERTED
    )
    update_data_config(settings)
    assert (
        settings["train_settings"]["data"]["tokenization"]
        == DINGO_T1_TOKENIZATION_CONVERTED
    )


def test_update_data_config_fills_dingo_t1_constant_defaults():
    """Constant defaults the old training code applied are filled in; bounds that it
    defaulted from the domain stay absent."""
    data_settings = {
        "detectors": ["H1", "L1"],
        "tokenization": {
            "num_tokens": 40,
            "drop_frequency_range": {"f_cut": {}, "mask_interval": {"max_width": 5.0}},
            "drop_random_tokens": {"increase_p_until_epoch": 10},
        },
    }
    with pytest.warns(UserWarning, match="increase_p_until_epoch"):
        update_data_config({"train_settings": {"data": data_settings}})
    assert data_settings["tokenization"] == {
        "num_tokens_per_block": 40,
        "mask_frequency_range": {
            "p_mask": 0.2,
            "p_same_all_detectors": 0.2,
            "p_lower_upper_both": [0.4, 0.4, 0.2],
        },
        "mask_frequency_notches": {"p_per_detector": 0.2, "max_width": 5.0},
        "mask_random_tokens": {"p_mask": 0.4, "max_num_tokens": 40},
    }


def test_update_data_config_rejects_normalized_positions():
    settings = {
        "train_settings": {
            "data": {
                "detectors": ["H1"],
                "tokenization": {"normalize_frequency_for_positional_encoding": True},
            }
        }
    }
    with pytest.raises(NotImplementedError, match="normalize_frequency"):
        update_data_config(settings)


def test_update_data_config_noop_without_tokenization_or_data():
    for settings in (
        {"train_settings": {"data": {"detectors": ["H1"]}}},
        {"train_settings": {}},
        {},
    ):
        before = str(settings)
        update_data_config(settings)
        assert str(settings) == before
