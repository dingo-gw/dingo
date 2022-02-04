import pytest
import types
import torch
import torch.optim as optim
from dingo.core.nn.nsf import create_nsf_model, FlowWrapper, \
    create_nsf_with_rb_projection_embedding_net
from dingo.core.nn.enets import \
    create_enet_with_projection_layer_and_dense_resnet
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
        'input_dims': (2, 3, 8033),
        # 'n_rb': 200,
        'svd': {'size': 200},
        'V_rb_list': None,
        'output_dim': 128,
        'hidden_dims': [1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512,
                        512, 512, 256, 256, 256, 256, 256, 256, 128, 128, 128,
                        128, 128, 128],
        'activation': 'elu',
        'dropout': 0.0,
        'batch_norm': True,
        'added_context': True,
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
        'input_dims': (2, 3, 20),
        # 'n_rb': 10,
        'V_rb_list': None,
        'output_dim': 8,
        'hidden_dims': [32, 16, 8],
        'activation': 'elu',
        'dropout': 0.0,
        'batch_norm': True,
        'added_context': True,
        'svd': {'size': 10},
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
    d.x = torch.rand((d.batch_size, *d.embedding_net_kwargs['input_dims']))
    d.z = torch.ones((d.batch_size,
                      d.context_dim - d.embedding_net_kwargs['output_dim']))
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
    assert num_params == 131448775, 'Unexpected number of model parameters.'


def test_sample_method_of_nsf(data_setup_nsf_small):
    """
    Test the forward pass of flow model for log_prob.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net)

    samples = model.sample(d.x, d.z)

    assert samples.shape == d.y.shape, 'Unexpected shape of samples.'
    assert torch.all(samples > -10) and torch.all(samples < 10), \
        'Unexpected samples encountered. Network initialization or ' \
        'normalization seems broken.'

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

    loss = - model(d.y, d.x, d.z)
    assert list(loss.shape) == [d.batch_size], 'Unexpected output shape.'
    assert torch.all(loss > 0) and torch.all(loss < 40), \
        'Unexpected log prob encountered. Network initialization or ' \
        'normalization seems broken.'

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
        loss = - torch.mean(model(yy, xx, d.zz))
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    samples_n = torch.mean(model.sample(d.xx, d.zz, num_samples=100), axis=1)

    assert losses[-1] < losses[0], 'Loss did not improve in training.'
    assert torch.mean((torch.abs(samples_n - d.yy) > 0.8).float()) < 0.3, \
        'Training may not have worked. Check manually that sampling improves.'


def test_model_builder_for_nsf_with_rb_embedding_net(data_setup_nsf_small):
    """
    Test the builder function create_nsf_with_rb_projection_embedding_net.
    """

    d = data_setup_nsf_small

    model = create_nsf_with_rb_projection_embedding_net(d.nde_kwargs,
                                                        d.embedding_net_kwargs)

    loss = - model(d.y, d.x, d.z)
    assert list(loss.shape) == [d.batch_size], 'Unexpected output shape.'
    assert torch.all(loss > 0) and torch.all(loss < 40), \
        'Unexpected log prob encountered. Network initialization or ' \
        'normalization seems broken.'

    with pytest.raises(ValueError):
        model(d.y, d.z, d.x)
    with pytest.raises(ValueError):
        model(d.y, d.x, d.z, d.z)
    with pytest.raises(RuntimeError):
        model(d.y, d.x, d.x)
