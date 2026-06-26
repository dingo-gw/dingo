import copy

import pytest
import torch
from torch.nn import functional as F

from dingo.core.nn.resnet import DenseResidualNet, LinearLayer
from dingo.core.nn.transformer import Tokenizer, TransformerModel, create_transformer_enet


NUM_TOKENS = 6
NUM_FEATURES = 12
NUM_BLOCKS = 2
OUTPUT_DIM = 8


def make_tokenizer(num_blocks=NUM_BLOCKS, layer_norm=False, batch_norm=False):
    return Tokenizer(
        input_dims=[NUM_TOKENS, NUM_FEATURES],
        hidden_dims=[16, 16],
        output_dim=OUTPUT_DIM,
        activation=F.elu,
        num_blocks=num_blocks,
        layer_norm=layer_norm,
        batch_norm=batch_norm,
    )


def make_position(batch_size, num_tokens=NUM_TOKENS, num_blocks=NUM_BLOCKS):
    """position[..., 0]=f_min, position[..., 1]=f_max, position[..., 2]=detector idx."""
    f_min = torch.rand(batch_size, num_tokens)
    f_max = f_min + torch.rand(batch_size, num_tokens)
    detector = torch.randint(0, num_blocks, (batch_size, num_tokens)).float()
    return torch.stack([f_min, f_max, detector], dim=-1)


def test_output_shape_batched():
    tokenizer = make_tokenizer()
    x = torch.rand(10, NUM_TOKENS, NUM_FEATURES)
    position = make_position(batch_size=10)
    out = tokenizer(x, position)
    assert out.shape == (10, NUM_TOKENS, OUTPUT_DIM)


def test_output_shape_unbatched():
    """No leading batch dimension, exercising DenseResidualNet's support for an
    arbitrary number of leading dims."""
    tokenizer = make_tokenizer()
    x = torch.rand(NUM_TOKENS, NUM_FEATURES)
    f_min = torch.rand(NUM_TOKENS)
    f_max = f_min + torch.rand(NUM_TOKENS)
    detector = torch.randint(0, NUM_BLOCKS, (NUM_TOKENS,)).float()
    position = torch.stack([f_min, f_max, detector], dim=-1)
    out = tokenizer(x, position)
    assert out.shape == (NUM_TOKENS, OUTPUT_DIM)


def test_invalid_input_dims_raises():
    with pytest.raises(ValueError):
        Tokenizer(
            input_dims=[NUM_TOKENS, NUM_FEATURES, 1],
            hidden_dims=[16],
            output_dim=OUTPUT_DIM,
            activation=F.elu,
            num_blocks=NUM_BLOCKS,
        )


def test_wrong_feature_dim_raises():
    tokenizer = make_tokenizer()
    x = torch.rand(10, NUM_TOKENS, NUM_FEATURES + 1)
    position = make_position(batch_size=10)
    with pytest.raises(ValueError):
        tokenizer(x, position)


def test_position_affects_output():
    """Changing f_min/f_max (with x and detector held fixed) must change the output,
    since the tokenizer is conditioned on position."""
    tokenizer = make_tokenizer()
    tokenizer.eval()
    x = torch.rand(5, NUM_TOKENS, NUM_FEATURES)
    position = make_position(batch_size=5)

    out_reference = tokenizer(x, position)

    position_modified = position.clone()
    position_modified[..., 0] += 1.0  # shift f_min
    out_modified = tokenizer(x, position_modified)

    assert not torch.allclose(out_reference, out_modified)


def test_detector_one_hot_distinguishes_tokens():
    """Two tokens with identical features and f_min/f_max but different detector
    indices must produce different embeddings."""
    tokenizer = make_tokenizer(num_blocks=2)
    tokenizer.eval()
    x = torch.rand(1, 2, NUM_FEATURES).expand(1, 2, NUM_FEATURES).clone()
    x[:, 1, :] = x[:, 0, :]  # identical features for both tokens

    f_min = torch.tensor([[0.5, 0.5]])
    f_max = torch.tensor([[0.8, 0.8]])
    detector = torch.tensor([[0.0, 1.0]])  # only detector index differs
    position = torch.stack([f_min, f_max, detector], dim=-1)

    out = tokenizer(x, position)
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])


def test_position_does_not_mix_across_tokens():
    """Regression test mirroring the MyResidualBlock GLU fix: changing one token's
    position must not affect another token's output."""
    tokenizer = make_tokenizer()
    tokenizer.eval()
    x = torch.rand(4, NUM_TOKENS, NUM_FEATURES)
    position = make_position(batch_size=4)

    out_reference = tokenizer(x, position)

    position_modified = position.clone()
    position_modified[:, 1, :] = make_position(batch_size=4)[:, 1, :]
    out_modified = tokenizer(x, position_modified)

    assert torch.allclose(out_reference[:, 0, :], out_modified[:, 0, :])
    assert torch.allclose(out_reference[:, 2:, :], out_modified[:, 2:, :])
    assert not torch.allclose(out_reference[:, 1, :], out_modified[:, 1, :])


def test_backward_pass():
    tokenizer = make_tokenizer(layer_norm=True)
    x = torch.rand(8, NUM_TOKENS, NUM_FEATURES)
    position = make_position(batch_size=8)
    target = torch.rand(8, NUM_TOKENS, OUTPUT_DIM)
    loss_fn = torch.nn.L1Loss()

    out_0 = tokenizer(x, position)
    loss_before = loss_fn(out_0, target)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=0.001)
    loss_before.backward()
    optimizer.step()

    out_1 = tokenizer(x, position)
    loss_after = loss_fn(out_1, target)
    assert loss_after < loss_before


# ---------------------------------------------------------------------------
# create_transformer_enet
# ---------------------------------------------------------------------------

D_MODEL = 16


def make_enet_kwargs():
    tokenizer_kwargs = {
        "input_dims": [NUM_TOKENS, NUM_FEATURES],
        "num_blocks": NUM_BLOCKS,
        "hidden_dims": [16],
        "activation": "elu",
        "batch_norm": False,
        "layer_norm": True,
    }
    transformer_kwargs = {
        "d_model": D_MODEL,
        "dim_feedforward": 32,
        "nhead": 4,
        "dropout": 0.0,
        "num_layers": 2,
        "norm_first": True,
    }
    return tokenizer_kwargs, transformer_kwargs


def make_enet_inputs(batch_size):
    x = torch.rand(batch_size, NUM_TOKENS, NUM_FEATURES)
    position = make_position(batch_size=batch_size)
    return x, position


def test_create_transformer_enet_default_pooling_is_cls():
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    model = create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs, transformer_kwargs=transformer_kwargs
    )
    assert model.pooling == "cls"
    assert hasattr(model, "class_token")


@pytest.mark.parametrize("pooling", ["cls", "average"])
def test_create_transformer_enet_without_final_net(pooling):
    """If final_net_kwargs is None, the pooled d_model-dim vector is returned as is."""
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    model = create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        pooling=pooling,
    )
    assert model.final_net is None

    x, position = make_enet_inputs(batch_size=5)
    out = model(x=x, position=position)
    assert out.shape == (5, D_MODEL)


def test_create_transformer_enet_with_linear_final_net():
    """final_net_kwargs without hidden_dims builds a LinearLayer."""
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    final_net_kwargs = {"activation": "elu", "output_dim": 5}
    model = create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        final_net_kwargs=final_net_kwargs,
    )
    assert isinstance(model.final_net, LinearLayer)

    x, position = make_enet_inputs(batch_size=5)
    out = model(x=x, position=position)
    assert out.shape == (5, 5)


def test_create_transformer_enet_with_dense_residual_final_net():
    """final_net_kwargs with hidden_dims builds a DenseResidualNet."""
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    final_net_kwargs = {
        "activation": "elu",
        "output_dim": 5,
        "hidden_dims": [8, 8],
        "layer_norm": True,
    }
    model = create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        final_net_kwargs=final_net_kwargs,
    )
    assert isinstance(model.final_net, DenseResidualNet)

    x, position = make_enet_inputs(batch_size=5)
    out = model(x=x, position=position)
    assert out.shape == (5, 5)


def test_create_transformer_enet_does_not_mutate_input_kwargs():
    """Settings dicts (e.g., loaded once from yaml and reused across training stages)
    must not be mutated by repeated calls."""
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    final_net_kwargs = {"activation": "elu", "output_dim": 5}
    tokenizer_kwargs_ref = copy.deepcopy(tokenizer_kwargs)
    final_net_kwargs_ref = copy.deepcopy(final_net_kwargs)

    create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        final_net_kwargs=final_net_kwargs,
    )
    # second call with the same (unmutated) dicts must not raise
    create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        final_net_kwargs=final_net_kwargs,
    )

    assert tokenizer_kwargs == tokenizer_kwargs_ref
    assert final_net_kwargs == final_net_kwargs_ref


def test_invalid_pooling_raises():
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    with pytest.raises(ValueError, match="pooling"):
        create_transformer_enet(
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
            pooling="max",
        )


# ---------------------------------------------------------------------------
# TransformerModel — src_key_padding_mask (drop-token masking)
# ---------------------------------------------------------------------------


def make_full_enet(pooling="cls"):
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    final_net_kwargs = {"activation": "elu", "output_dim": 5}
    return create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        pooling=pooling,
        final_net_kwargs=final_net_kwargs,
    )


def test_padding_mask_does_not_change_output_shape():
    """Output shape must be identical whether or not a padding mask is supplied."""
    model = make_full_enet(pooling="cls")
    model.eval()
    x, position = make_enet_inputs(batch_size=4)
    mask = torch.zeros(4, NUM_TOKENS, dtype=torch.bool)
    mask[:, -2:] = True  # mask last two tokens

    out_no_mask = model(x=x, position=position)
    out_masked = model(x=x, position=position, src_key_padding_mask=mask)
    assert out_no_mask.shape == out_masked.shape


def test_padding_mask_changes_output():
    """Masking some tokens must change the CLS-pooled output."""
    model = make_full_enet(pooling="cls")
    model.eval()
    x, position = make_enet_inputs(batch_size=4)
    mask = torch.zeros(4, NUM_TOKENS, dtype=torch.bool)
    mask[:, -2:] = True

    out_no_mask = model(x=x, position=position)
    out_masked = model(x=x, position=position, src_key_padding_mask=mask)
    assert not torch.allclose(out_no_mask, out_masked)


def test_average_pooling_ignores_fully_masked_token():
    """For average pooling, a fully masked token should not affect the result."""
    model = make_full_enet(pooling="average")
    model.eval()
    x, position = make_enet_inputs(batch_size=2)

    # mask with last token dropped
    mask_drop = torch.zeros(2, NUM_TOKENS, dtype=torch.bool)
    mask_drop[:, -1] = True

    out_drop = model(x=x, position=position, src_key_padding_mask=mask_drop)

    # Replace the last token's features with noise — output should be unchanged
    x_noisy = x.clone()
    x_noisy[:, -1, :] = torch.rand_like(x_noisy[:, -1, :]) * 1e3
    out_noisy = model(x=x_noisy, position=position, src_key_padding_mask=mask_drop)

    assert torch.allclose(out_drop, out_noisy, atol=1e-5)


def test_create_transformer_enet_backward_pass():
    tokenizer_kwargs, transformer_kwargs = make_enet_kwargs()
    final_net_kwargs = {"activation": "elu", "output_dim": 5}
    model = create_transformer_enet(
        tokenizer_kwargs=tokenizer_kwargs,
        transformer_kwargs=transformer_kwargs,
        final_net_kwargs=final_net_kwargs,
    )

    x, position = make_enet_inputs(batch_size=8)
    target = torch.rand(8, 5)
    loss_fn = torch.nn.L1Loss()

    out_0 = model(x=x, position=position)
    loss_before = loss_fn(out_0, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_before.backward()
    optimizer.step()

    out_1 = model(x=x, position=position)
    loss_after = loss_fn(out_1, target)
    assert loss_after < loss_before
