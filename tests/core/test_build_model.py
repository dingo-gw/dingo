import copy

import torch

from dingo.core.posterior_models.build_model import autocomplete_model_kwargs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_PARAMS = 4
_CONTEXT_DIM = 8
_D_MODEL = 16
_NUM_TOKENS = 6
_NUM_FEATURES = 12
_NUM_BLOCKS = 2


def _make_data_sample(
    num_params=_NUM_PARAMS, waveform_shape=(_NUM_TOKENS, _NUM_FEATURES)
):
    parameters = torch.zeros(num_params)
    waveform = torch.zeros(*waveform_shape)
    return [parameters, waveform]


def _make_transformer_model_kwargs(
    final_net_output_dim=_CONTEXT_DIM, include_final_net=True
):
    kwargs = {
        "embedding_type": "transformer",
        "posterior_kwargs": {"input_dim": None, "context_dim": None},
        "embedding_kwargs": {
            "tokenizer_kwargs": {
                "num_blocks": _NUM_BLOCKS,
                "hidden_dims": [16],
                "activation": "elu",
            },
            "transformer_kwargs": {"d_model": _D_MODEL},
            "pooling": "cls",
        },
    }
    if include_final_net:
        kwargs["embedding_kwargs"]["final_net_kwargs"] = {
            "activation": "elu",
            "output_dim": final_net_output_dim,
        }
    return kwargs


def _make_resnet_model_kwargs(output_dim=_CONTEXT_DIM):
    return {
        "embedding_type": "resnet",
        "posterior_kwargs": {"input_dim": None, "context_dim": None},
        "embedding_kwargs": {
            "output_dim": output_dim,
            "hidden_dims": [32],
        },
    }


# ---------------------------------------------------------------------------
# autocomplete_model_kwargs — transformer path
# ---------------------------------------------------------------------------


def test_autocomplete_transformer_sets_tokenizer_input_dims():
    model_kwargs = _make_transformer_model_kwargs()
    data_sample = _make_data_sample()

    autocomplete_model_kwargs(model_kwargs, data_sample)

    assert model_kwargs["embedding_kwargs"]["tokenizer_kwargs"]["input_dims"] == list(
        data_sample[1].shape
    )


def test_autocomplete_transformer_context_dim_from_final_net():
    model_kwargs = _make_transformer_model_kwargs(final_net_output_dim=_CONTEXT_DIM)
    autocomplete_model_kwargs(model_kwargs, _make_data_sample())
    assert model_kwargs["posterior_kwargs"]["context_dim"] == _CONTEXT_DIM


def test_autocomplete_transformer_context_dim_from_d_model():
    """When final_net_kwargs is absent, context_dim falls back to transformer d_model."""
    model_kwargs = _make_transformer_model_kwargs(include_final_net=False)
    autocomplete_model_kwargs(model_kwargs, _make_data_sample())
    assert model_kwargs["posterior_kwargs"]["context_dim"] == _D_MODEL


def test_autocomplete_transformer_sets_input_dim():
    model_kwargs = _make_transformer_model_kwargs()
    data_sample = _make_data_sample(num_params=7)
    autocomplete_model_kwargs(model_kwargs, data_sample)
    assert model_kwargs["posterior_kwargs"]["input_dim"] == 7


def test_autocomplete_transformer_infers_num_blocks_from_position():
    model_kwargs = _make_transformer_model_kwargs()
    # position tensor: [num_tokens, 3], column 2 = detector index (0 or 1 → num_blocks=2)
    f_min = torch.rand(_NUM_TOKENS)
    f_max = f_min + 0.1
    detector = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    position = torch.stack([f_min, f_max, detector], dim=-1)
    data_sample = _make_data_sample() + [position]
    autocomplete_model_kwargs(model_kwargs, data_sample)
    assert model_kwargs["embedding_kwargs"]["tokenizer_kwargs"]["num_blocks"] == 2


def test_autocomplete_transformer_preserves_explicit_num_blocks():
    model_kwargs = _make_transformer_model_kwargs()
    model_kwargs["embedding_kwargs"]["tokenizer_kwargs"]["num_blocks"] = 3
    f_min = torch.rand(_NUM_TOKENS)
    detector = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    position = torch.stack([f_min, f_min + 0.1, detector], dim=-1)
    data_sample = _make_data_sample() + [position]
    autocomplete_model_kwargs(model_kwargs, data_sample)
    assert model_kwargs["embedding_kwargs"]["tokenizer_kwargs"]["num_blocks"] == 3


def test_autocomplete_transformer_does_not_set_added_context():
    """added_context is a resnet-only concept; it must not appear in transformer kwargs."""
    model_kwargs = _make_transformer_model_kwargs()
    autocomplete_model_kwargs(model_kwargs, _make_data_sample())
    assert "added_context" not in model_kwargs["embedding_kwargs"]


# ---------------------------------------------------------------------------
# autocomplete_model_kwargs — resnet path (regression)
# ---------------------------------------------------------------------------


def test_autocomplete_resnet_sets_input_dims():
    raw_waveform_shape = (2, 3, 20)
    model_kwargs = _make_resnet_model_kwargs()
    data_sample = _make_data_sample(waveform_shape=raw_waveform_shape)

    autocomplete_model_kwargs(model_kwargs, data_sample)

    assert model_kwargs["embedding_kwargs"]["input_dims"] == list(raw_waveform_shape)


def test_autocomplete_resnet_sets_context_dim():
    model_kwargs = _make_resnet_model_kwargs(output_dim=10)
    autocomplete_model_kwargs(model_kwargs, _make_data_sample())
    assert model_kwargs["posterior_kwargs"]["context_dim"] == 10


def test_autocomplete_resnet_sets_added_context_false_without_gnpe():
    model_kwargs = _make_resnet_model_kwargs()
    data_sample = _make_data_sample()  # only 2 elements, no GNPE proxies
    autocomplete_model_kwargs(model_kwargs, data_sample)
    assert model_kwargs["embedding_kwargs"]["added_context"] is False


def test_autocomplete_resnet_sets_added_context_true_with_gnpe():
    model_kwargs = _make_resnet_model_kwargs(output_dim=8)
    gnpe_proxies = torch.zeros(3)
    data_sample = _make_data_sample() + [gnpe_proxies]
    autocomplete_model_kwargs(model_kwargs, data_sample)
    assert model_kwargs["embedding_kwargs"]["added_context"] is True
    assert model_kwargs["posterior_kwargs"]["context_dim"] == 8 + 3


def test_autocomplete_does_not_mutate_data_sample():
    model_kwargs = _make_transformer_model_kwargs()
    data_sample = _make_data_sample()
    data_sample_ref = [t.clone() for t in data_sample]
    autocomplete_model_kwargs(model_kwargs, data_sample)
    for original, ref in zip(data_sample, data_sample_ref):
        assert torch.equal(original, ref)
