import pytest
import torch
from torch.nn import functional as F

from dingo.core.nn.resnet import DenseResidualNet, LinearLayer, MyResidualBlock
from testutils_enets import check_model_forward_pass, check_model_backward_pass


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


def test_forward_pass_of_DenseResidualNet():
    """Forward pass with plain 2D [batch, features] input."""
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_forward_pass(enet, [output_dim], [input_dim], batch_size)


def test_backward_pass_of_DenseResidualNet():
    """Backward pass / optimizer step with plain 2D [batch, features] input."""
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_backward_pass(enet, [input_dim], batch_size)


def test_forward_pass_with_3d_input():
    """Forward pass with token-batched [batch, tokens, features] input, as used by
    the transformer tokenizer. batch_norm is disabled since nn.BatchNorm1d treats
    dim 1 as the channel axis, which for 3D input is the token axis, not features;
    only layer_norm supports 3D input."""
    batch_size, num_tokens = 100, 7
    input_dim, output_dim, hidden_dims = 120, 8, (64, 32, 64)
    enet = DenseResidualNet(
        input_dim, output_dim, hidden_dims, batch_norm=False, layer_norm=True
    )
    x = torch.rand(batch_size, num_tokens, input_dim)
    y = enet(x)
    assert y.shape == (batch_size, num_tokens, output_dim)


def test_layer_norm_and_batch_norm_are_mutually_exclusive():
    with pytest.raises(ValueError):
        MyResidualBlock(features=16, use_batch_norm=True, use_layer_norm=True)


def test_context_glu_does_not_mix_across_tokens():
    """Regression test for the dim=1 -> dim=-1 GLU fix.

    With 3D [batch, tokens, features] input, each token's output must depend only on
    its own context, not on other tokens' context. The original glasflow ResidualBlock
    used dim=1 for the GLU, which for 3D input operates over the token axis instead of
    the feature axis, leaking context across tokens.
    """
    torch.manual_seed(0)
    features, context_features, num_tokens, batch_size = 16, 4, 3, 5
    block = MyResidualBlock(features=features, context_features=context_features)
    block.eval()

    x = torch.rand(batch_size, num_tokens, features)
    context = torch.rand(batch_size, num_tokens, context_features)

    out_reference = block(x, context=context)

    # Change only the context of token 1; token 0 and token 2 outputs must be
    # unaffected if the GLU correctly operates per-token along the feature axis.
    context_modified = context.clone()
    context_modified[:, 1, :] = torch.rand(batch_size, context_features)
    out_modified = block(x, context=context_modified)

    assert torch.allclose(out_reference[:, 0, :], out_modified[:, 0, :])
    assert torch.allclose(out_reference[:, 2, :], out_modified[:, 2, :])
    assert not torch.allclose(out_reference[:, 1, :], out_modified[:, 1, :])


def test_dense_residual_net_exposes_hidden_features_for_uniform_widths():
    """nflows' coupling transforms scale spline parameters by sqrt(hidden_features)
    iff the conditioner has the attribute; removing it changes trained flows."""
    from torch.nn import functional as F
    from dingo.core.nn.resnet import DenseResidualNet

    net = DenseResidualNet(4, 6, (8, 8), activation=F.relu)
    assert net.hidden_features == 8
    assert not hasattr(
        DenseResidualNet(4, 6, (8, 16), activation=F.relu), "hidden_features"
    )
