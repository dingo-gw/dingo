import pytest
import torch
import torch.nn as nn
from glasflow.nflows.nn.nets.resnet import ResidualBlock

from dingo.core.nn.resnet import DenseResidualNet, MyResidualBlock


@pytest.mark.parametrize("norm", ["BatchNorm", "LayerNorm", None])
def test_dense_residual_net_forward(norm):
    net = DenseResidualNet(
        input_dim=6, output_dim=3, hidden_dims=(8, 8, 4), norm=norm, context_features=2
    )
    out = net(torch.randn(5, 6), context=torch.randn(5, 2))
    assert out.shape == (5, 3)


def test_batch_norm_state_dict_matches_nflows():
    """
    With BatchNorm the parameter names and eps must match the nflows
    ResidualBlock, so that existing checkpoints remain loadable.
    """
    torch.manual_seed(0)
    ref = ResidualBlock(features=8, context_features=2, use_batch_norm=True)
    block = MyResidualBlock(features=8, context_features=2, norm="BatchNorm")
    assert block.state_dict().keys() == ref.state_dict().keys()
    block.load_state_dict(ref.state_dict())
    assert all(bn.eps == 1e-3 for bn in block.batch_norm_layers)
    x, c = torch.randn(4, 8), torch.randn(4, 2)
    ref.eval(), block.eval()
    torch.testing.assert_close(block(x, c), ref(x, c))


def test_layer_norm_keys_and_eps():
    """LayerNorm parameters live under layer_norm_layers with the default eps."""
    block = MyResidualBlock(features=8, norm="LayerNorm")
    keys = block.state_dict().keys()
    assert any(k.startswith("layer_norm_layers.") for k in keys)
    assert not any(k.startswith("batch_norm_layers.") for k in keys)
    assert all(isinstance(ln, nn.LayerNorm) for ln in block.layer_norm_layers)
    assert all(ln.eps == nn.LayerNorm(8).eps for ln in block.layer_norm_layers)


def test_no_norm_has_no_norm_layers():
    block = MyResidualBlock(features=8, norm=None)
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)) for m in block.modules()
    )


def test_invalid_norm_option():
    with pytest.raises(ValueError, match="norm must be"):
        MyResidualBlock(features=8, norm="GroupNorm")
    with pytest.raises(ValueError, match="norm must be"):
        DenseResidualNet(4, 2, (8,), norm="GroupNorm")


def test_context_glu_does_not_mix_across_tokens():
    """
    With token-batched [batch, tokens, features] input, each token's output must
    depend only on its own context. The GLU has to act along the feature axis
    (dim=-1); glasflow's ResidualBlock uses dim=1, which for 3D input is the
    token axis and leaks context across tokens.
    """
    torch.manual_seed(0)
    block = MyResidualBlock(features=16, context_features=4, norm="LayerNorm")
    block.eval()
    x = torch.rand(5, 3, 16)
    context = torch.rand(5, 3, 4)
    out_reference = block(x, context=context)
    context_modified = context.clone()
    context_modified[:, 1, :] = torch.rand(5, 4)
    out_modified = block(x, context=context_modified)
    assert torch.allclose(out_reference[:, 0, :], out_modified[:, 0, :])
    assert torch.allclose(out_reference[:, 2, :], out_modified[:, 2, :])
    assert not torch.allclose(out_reference[:, 1, :], out_modified[:, 1, :])
