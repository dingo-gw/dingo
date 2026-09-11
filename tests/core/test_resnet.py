import pytest
import torch
import torch.nn as nn
from glasflow.nflows.nn.nets.resnet import ResidualBlock
from torch.nn import functional as F

from dingo.core.nn.resnet import DenseResidualNet, LinearLayer, MyResidualBlock
from testutils_enets import check_model_backward_pass, check_model_forward_pass


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


def test_forward_pass_of_LinearLayer():
    batch_size, input_dim, output_dim = 10, 16, 4
    layer = LinearLayer(input_dim=input_dim, output_dim=output_dim, activation=F.elu)
    check_model_forward_pass(layer, [output_dim], [input_dim], batch_size)


def test_LinearLayer_without_activation_is_a_bare_projection():
    layer = LinearLayer(input_dim=16, output_dim=4)
    x = torch.randn(10, 16)
    assert torch.equal(layer(x), layer.linear(x))


def test_backward_pass_of_LinearLayer():
    batch_size, input_dim, output_dim = 10, 16, 4
    layer = LinearLayer(input_dim=input_dim, output_dim=output_dim, activation=F.elu)
    check_model_backward_pass(layer, [input_dim], batch_size)


def test_backward_pass_of_DenseResidualNet():
    """Backward pass / optimizer step with plain 2D [batch, features] input."""
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_backward_pass(enet, [input_dim], batch_size)


def test_forward_pass_with_3d_input():
    """Forward pass with token-batched [batch, tokens, features] input, as used by
    the transformer token embedding. Only LayerNorm (or no normalization) supports
    3D input: nn.BatchNorm1d treats dim 1 as the channel axis, which for 3D input is
    the token axis, not features."""
    batch_size, num_tokens = 100, 7
    input_dim, output_dim, hidden_dims = 120, 8, (64, 32, 64)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims, norm="LayerNorm")
    x = torch.rand(batch_size, num_tokens, input_dim)
    y = enet(x)
    assert y.shape == (batch_size, num_tokens, output_dim)


def test_dense_residual_net_exposes_hidden_features_for_uniform_widths():
    """nflows' coupling transforms scale spline parameters by sqrt(hidden_features)
    iff the conditioner has the attribute; removing it changes trained flows."""
    net = DenseResidualNet(4, 6, (8, 8), activation=F.relu)
    assert net.hidden_features == 8
    assert not hasattr(
        DenseResidualNet(4, 6, (8, 16), activation=F.relu), "hidden_features"
    )
