import pytest
import torch
import torch.optim as optim

from dingo.core.nn.cfnets import create_cf


THETA_DIM = 14
CONTEXT_DIM = 32
INPUT_DIMS = (2, 3, 50)
BATCH_SIZE = 16


def _base_kwargs(batch_norm: bool, layer_norm: bool):
    posterior_kwargs = {
        "input_dim": THETA_DIM,
        "context_dim": CONTEXT_DIM,
        "activation": "gelu",
        "batch_norm": batch_norm,
        "layer_norm": layer_norm,
        "dropout": 0.0,
        "hidden_dims": [64, 32, 16],
        "sigma_min": 0.001,
        "theta_embedding_kwargs": {
            "embedding_net": {
                "activation": "gelu",
                "hidden_dims": [16, 32],
                "output_dim": 8,
                "batch_norm": batch_norm,
                "layer_norm": layer_norm,
            },
            "encoding": {"encode_all": False, "frequencies": 0},
        },
        "theta_with_glu": True,
        "context_with_glu": False,
    }
    embedding_kwargs = {
        "input_dims": INPUT_DIMS,
        "V_rb_list": None,
        "output_dim": CONTEXT_DIM,
        "hidden_dims": [32, 16],
        "activation": "gelu",
        "batch_norm": batch_norm,
        "layer_norm": layer_norm,
        "svd": {"size": 10},
    }
    return posterior_kwargs, embedding_kwargs


@pytest.mark.parametrize(
    "batch_norm,layer_norm", [(True, False), (False, True), (False, False)]
)
def test_create_cf_forward_pass(batch_norm, layer_norm):
    """Forward pass of the continuous flow network built by create_cf, exercising
    both the continuous_flow_net and the theta/context DenseResidualNet embeddings
    with batch_norm, layer_norm, and neither.
    """
    posterior_kwargs, embedding_kwargs = _base_kwargs(batch_norm, layer_norm)
    model = create_cf(posterior_kwargs, embedding_kwargs)

    t = torch.rand(BATCH_SIZE)
    theta = torch.rand(BATCH_SIZE, THETA_DIM)
    context = torch.rand(BATCH_SIZE, *INPUT_DIMS)

    out = model(t, theta, context)
    assert out.shape == (BATCH_SIZE, THETA_DIM)


@pytest.mark.parametrize(
    "batch_norm,layer_norm", [(True, False), (False, True), (False, False)]
)
def test_create_cf_backward_pass(batch_norm, layer_norm):
    """Backward pass and optimizer step of the continuous flow network built by
    create_cf: checks that gradients reach all three DenseResidualNet instances
    (continuous_flow_net, context embedding, theta embedding) and that the loss
    decreases after an optimizer step.
    """
    posterior_kwargs, embedding_kwargs = _base_kwargs(batch_norm, layer_norm)
    model = create_cf(posterior_kwargs, embedding_kwargs)

    t = torch.rand(BATCH_SIZE)
    theta = torch.rand(BATCH_SIZE, THETA_DIM)
    context = torch.rand(BATCH_SIZE, *INPUT_DIMS)
    target = torch.rand(BATCH_SIZE, THETA_DIM)
    loss_fn = torch.nn.L1Loss()

    out_0 = model(t, theta, context)
    loss_before = loss_fn(out_0, target)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_before.backward()

    assert model.continuous_flow_net.initial_layer.weight.grad is not None
    assert model.context_embedding_net[1].initial_layer.weight.grad is not None
    assert model.theta_embedding_net[1].initial_layer.weight.grad is not None

    optimizer.step()
    out_1 = model(t, theta, context)
    loss_after = loss_fn(out_1, target)
    assert loss_after < loss_before


def test_create_cf_without_context_embedding():
    """When embedding_kwargs is None, the context embedding is an identity mapping
    and the continuous flow net consumes the raw context directly."""
    posterior_kwargs, _ = _base_kwargs(batch_norm=False, layer_norm=True)
    posterior_kwargs["context_dim"] = CONTEXT_DIM
    model = create_cf(posterior_kwargs, embedding_kwargs=None)

    t = torch.rand(BATCH_SIZE)
    theta = torch.rand(BATCH_SIZE, THETA_DIM)
    context = torch.rand(BATCH_SIZE, CONTEXT_DIM)

    out = model(t, theta, context)
    assert out.shape == (BATCH_SIZE, THETA_DIM)
