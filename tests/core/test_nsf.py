import copy
import pytest
import types
import torch
import torch.optim as optim
from dingo.core.nn.nsf import (
    create_nsf_model,
    create_nsf_with_rb_projection_embedding_net,
    create_nsf_with_transformer_embedding_net,
    create_transform,
    FlowWrapper,
)
from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet
from dingo.core.utils import torchutils


@pytest.fixture()
def data_setup_nsf_large():
    d = types.SimpleNamespace()
    d.input_dim = 15
    d.context_dim = 129
    d.num_flow_steps = 30
    d.base_transform_kwargs = {
        "hidden_dim": 512,
        "num_transform_blocks": 5,
        "activation": "elu",
        "dropout_probability": 0.0,
        "batch_norm": True,
        "num_bins": 8,
        "base_transform_type": "rq-coupling",
    }
    d.embedding_net_kwargs = {
        "input_dims": (2, 3, 8033),
        # 'n_rb': 200,
        "svd": {"size": 200},
        "V_rb_list": None,
        "output_dim": 128,
        "hidden_dims": [
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            512,
            512,
            512,
            512,
            512,
            512,
            256,
            256,
            256,
            256,
            256,
            256,
            128,
            128,
            128,
            128,
            128,
            128,
        ],
        "activation": "elu",
        "dropout": 0.0,
        "batch_norm": True,
        "added_context": True,
    }
    d.embedding_net_builder = create_enet_with_projection_layer_and_dense_resnet
    d.nde_builder = create_nsf_model
    d.nde_kwargs = {
        "input_dim": d.input_dim,
        "context_dim": d.context_dim,
        "num_flow_steps": d.num_flow_steps,
        "base_transform_kwargs": d.base_transform_kwargs,
    }
    return d


@pytest.fixture()
def data_setup_nsf_small():
    d = types.SimpleNamespace()
    d.input_dim = 4
    d.context_dim = 10
    d.num_flow_steps = 5
    d.base_transform_kwargs = {
        "hidden_dim": 64,
        "num_transform_blocks": 2,
        "activation": "elu",
        "dropout_probability": 0.0,
        "batch_norm": True,
        "num_bins": 8,
        "base_transform_type": "rq-coupling",
    }
    d.embedding_net_kwargs = {
        "input_dims": (2, 3, 20),
        # 'n_rb': 10,
        "V_rb_list": None,
        "output_dim": 8,
        "hidden_dims": [32, 16, 8],
        "activation": "elu",
        "dropout": 0.0,
        "batch_norm": True,
        "added_context": True,
        "svd": {"size": 10},
    }
    d.embedding_net_builder = create_enet_with_projection_layer_and_dense_resnet
    d.nde_builder = create_nsf_model
    d.nde_kwargs = {
        "input_dim": d.input_dim,
        "context_dim": d.context_dim,
        "num_flow_steps": d.num_flow_steps,
        "base_transform_kwargs": d.base_transform_kwargs,
    }

    d.batch_size = 20
    d.x = torch.rand((d.batch_size, *d.embedding_net_kwargs["input_dims"]))
    d.z = torch.ones(
        (d.batch_size, d.context_dim - d.embedding_net_kwargs["output_dim"])
    )
    d.y = torch.ones((d.batch_size, d.input_dim))

    # build d.yy, which depends on input d.zz
    d.xx = torch.cat((d.x, d.x))
    d.yy = torch.cat((d.y, -d.y))
    d.zz = torch.cat((d.z, -d.z))

    return d


def test_nsf_number_of_parameters(data_setup_nsf_large):
    """
    Builds a neural spline flow with the hyperparameters from that used in
    https://arxiv.org/abs/2106.12594, and checks that the number of
    parameters matches the expected one.
    """

    d = data_setup_nsf_large

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net)

    num_params = torchutils.get_number_of_model_parameters(model)
    assert num_params == 131448775, "Unexpected number of model parameters."


def test_sample_method_of_nsf(data_setup_nsf_small):
    """
    Test the forward pass of flow model for log_prob.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net)

    samples = model.sample(d.x, d.z)
    # model.sample(num_samples=1) adds an extra dimension that needs to be squeezed.
    samples = samples.squeeze(1)

    assert samples.shape == d.y.shape, "Unexpected shape of samples."
    assert torch.all(samples > -10) and torch.all(samples < 10), (
        "Unexpected samples encountered. Network initialization or "
        "normalization seems broken."
    )

    with pytest.raises(ValueError):
        model.sample(d.z, d.x)
    with pytest.raises(ValueError):
        model.sample(d.x, d.z, d.z)
    with pytest.raises(RuntimeError):
        model.sample(d.x, d.x)


def test_forward_pass_for_log_prob_of_nsf(data_setup_nsf_small):
    """
    Test the forward pass of flow model for log_prob.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net)

    loss = -model(d.y, d.x, d.z)
    assert list(loss.shape) == [d.batch_size], "Unexpected output shape."
    assert torch.all(loss > 0) and torch.all(loss < 40), (
        "Unexpected log prob encountered. Network initialization or "
        "normalization seems broken."
    )

    with pytest.raises(ValueError):
        model(d.y, d.z, d.x)
    with pytest.raises(ValueError):
        model(d.y, d.x, d.z, d.z)
    with pytest.raises(RuntimeError):
        model(d.y, d.x, d.x)


def test_backward_pass_for_log_prob_of_nsf(data_setup_nsf_small):
    """
    Test the backward pass of flow model for log_prob. This function also
    checks that conditional sampling improves during training.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Simple train loop. The learned parameters yy are strongly correlated
    # with the context information d.zz. We check that (i) the loss and (ii)
    # the posterior samples improve during training.
    losses = []
    for idx in range(40):
        yy = d.yy + 0.02 * torch.rand_like(d.yy)
        xx = torch.rand_like(d.xx)
        loss = -torch.mean(model(yy, xx, d.zz))
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    samples_n = torch.mean(model.sample(d.xx, d.zz, num_samples=100), axis=1)

    assert losses[-1] < losses[0], "Loss did not improve in training."
    assert (
        torch.mean((torch.abs(samples_n - d.yy) > 0.8).float()) < 0.3
    ), "Training may not have worked. Check manually that sampling improves."


def test_model_builder_for_nsf_with_rb_embedding_net(data_setup_nsf_small):
    """
    Test the builder function create_nsf_with_rb_projection_embedding_net.
    """

    d = data_setup_nsf_small

    model = create_nsf_with_rb_projection_embedding_net(
        d.nde_kwargs, d.embedding_net_kwargs
    )

    loss = -model(d.y, d.x, d.z)
    assert list(loss.shape) == [d.batch_size], "Unexpected output shape."
    assert torch.all(loss > 0) and torch.all(loss < 40), (
        "Unexpected log prob encountered. Network initialization or "
        "normalization seems broken."
    )

    with pytest.raises(ValueError):
        model(d.y, d.z, d.x)
    with pytest.raises(ValueError):
        model(d.y, d.x, d.z, d.z)
    with pytest.raises(RuntimeError):
        model(d.y, d.x, d.x)


# ---------------------------------------------------------------------------
# create_nsf_with_transformer_embedding_net
# ---------------------------------------------------------------------------

_NUM_TOKENS = 6
_NUM_FEATURES = 12
_NUM_BLOCKS = 2
_NUM_PARAMS = 4
_CONTEXT_DIM = 8
_D_MODEL = 16
_BATCH_SIZE = 10


def _make_transformer_posterior_kwargs():
    return {
        "input_dim": _NUM_PARAMS,
        "context_dim": _CONTEXT_DIM,
        "num_flow_steps": 5,
        "base_transform_kwargs": {
            "hidden_dim": 32,
            "num_transform_blocks": 2,
            "activation": "elu",
            "dropout_probability": 0.0,
            "batch_norm": False,
            "num_bins": 4,
            "base_transform_type": "rq-coupling",
        },
    }


def _make_transformer_embedding_kwargs():
    return {
        "tokenizer_kwargs": {
            "input_dims": [_NUM_TOKENS, _NUM_FEATURES],
            "num_blocks": _NUM_BLOCKS,
            "hidden_dims": [16],
            "activation": "elu",
            "batch_norm": False,
            "layer_norm": False,
        },
        "transformer_kwargs": {
            "d_model": _D_MODEL,
            "dim_feedforward": 32,
            "nhead": 4,
            "dropout": 0.0,
            "num_layers": 2,
            "norm_first": True,
        },
        "pooling": "cls",
        "final_net_kwargs": {
            "activation": "elu",
            "output_dim": _CONTEXT_DIM,
        },
    }


def _make_transformer_inputs(batch_size=_BATCH_SIZE):
    waveform = torch.rand(batch_size, _NUM_TOKENS, _NUM_FEATURES)
    f_min = torch.rand(batch_size, _NUM_TOKENS)
    f_max = f_min + torch.rand(batch_size, _NUM_TOKENS)
    detector = torch.randint(0, _NUM_BLOCKS, (batch_size, _NUM_TOKENS)).float()
    position = torch.stack([f_min, f_max, detector], dim=-1)
    theta = torch.randn(batch_size, _NUM_PARAMS)
    return theta, waveform, position


def test_nsf_with_transformer_enet_output_shape():
    model = create_nsf_with_transformer_embedding_net(
        posterior_kwargs=_make_transformer_posterior_kwargs(),
        embedding_kwargs=_make_transformer_embedding_kwargs(),
    )
    theta, waveform, position = _make_transformer_inputs()
    log_prob = model(theta, waveform, position)
    assert log_prob.shape == (_BATCH_SIZE,), "Unexpected log_prob shape."


def test_nsf_with_transformer_enet_backward_pass():
    model = create_nsf_with_transformer_embedding_net(
        posterior_kwargs=_make_transformer_posterior_kwargs(),
        embedding_kwargs=_make_transformer_embedding_kwargs(),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    theta, waveform, position = _make_transformer_inputs(batch_size=32)

    losses = []
    for _ in range(10):
        loss = -model(theta, waveform, position).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert losses[-1] < losses[0], "Loss did not decrease during training."


def test_nsf_with_transformer_enet_ignores_allow_tf32():
    """allow_tf32 is a legacy key that must be silently dropped."""
    embedding_kwargs = _make_transformer_embedding_kwargs()
    embedding_kwargs["allow_tf32"] = False
    model = create_nsf_with_transformer_embedding_net(
        posterior_kwargs=_make_transformer_posterior_kwargs(),
        embedding_kwargs=embedding_kwargs,
    )
    theta, waveform, position = _make_transformer_inputs()
    assert model(theta, waveform, position).shape == (_BATCH_SIZE,)


def test_nsf_with_transformer_enet_does_not_mutate_kwargs():
    posterior_kwargs = _make_transformer_posterior_kwargs()
    embedding_kwargs = _make_transformer_embedding_kwargs()
    posterior_kwargs_ref = copy.deepcopy(posterior_kwargs)
    embedding_kwargs_ref = copy.deepcopy(embedding_kwargs)

    create_nsf_with_transformer_embedding_net(
        posterior_kwargs=posterior_kwargs,
        embedding_kwargs=embedding_kwargs,
    )

    assert posterior_kwargs == posterior_kwargs_ref
    assert embedding_kwargs == embedding_kwargs_ref


# ---------------------------------------------------------------------------
# create_base_transform / create_transform — layer_norm=True path
# ---------------------------------------------------------------------------


def test_create_transform_layer_norm_output_shape():
    """create_transform with layer_norm=True in base_transform_kwargs must produce
    log-probs of the same shape as without layer_norm."""
    param_dim = 6
    context_dim = 8
    batch_size = 10

    base_transform_kwargs = {
        "hidden_dim": 16,
        "num_transform_blocks": 2,
        "activation": "elu",
        "dropout_probability": 0.0,
        "batch_norm": False,
        "layer_norm": True,
        "num_bins": 4,
        "base_transform_type": "rq-coupling",
    }
    transform = create_transform(
        num_flow_steps=3,
        param_dim=param_dim,
        context_dim=context_dim,
        base_transform_kwargs=base_transform_kwargs,
    )

    y = torch.randn(batch_size, param_dim)
    context = torch.randn(batch_size, context_dim)
    y_transformed, log_det = transform(y, context=context)

    assert y_transformed.shape == (batch_size, param_dim)
    assert log_det.shape == (batch_size,)


def test_create_transform_layer_norm_backward_pass():
    """Backward pass through a coupling flow with layer_norm=True must not error."""
    param_dim = 4
    context_dim = 6

    base_transform_kwargs = {
        "hidden_dim": 16,
        "num_transform_blocks": 2,
        "activation": "elu",
        "dropout_probability": 0.0,
        "batch_norm": False,
        "layer_norm": True,
        "num_bins": 4,
        "base_transform_type": "rq-coupling",
    }
    transform = create_transform(
        num_flow_steps=2,
        param_dim=param_dim,
        context_dim=context_dim,
        base_transform_kwargs=base_transform_kwargs,
    )

    y = torch.randn(8, param_dim, requires_grad=True)
    context = torch.randn(8, context_dim)
    _, log_det = transform(y, context=context)
    log_det.sum().backward()
    assert y.grad is not None
