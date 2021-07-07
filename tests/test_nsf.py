import pytest
import types
import torch
from dingo.core.nn.nsf import create_nsf_model, FlowWrapper
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
        'n_rb': 200,
        'V_rb_list': None,
        'output_dim': 128,
        'hidden_dims': [1024, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128],
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
    d.input_dim = 15
    d.context_dim = 130
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
        'input_dims': (2, 3, 200),
        'n_rb': 10,
        'V_rb_list': None,
        'output_dim': 32,
        'hidden_dims': [128, 64, 32],
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

    d.batch_size = 10
    d.x = torch.rand((d.batch_size, *d.embedding_net_kwargs['input_dims']))
    d.z = torch.ones((d.batch_size,
                      d.context_dim - d.embedding_net_kwargs['output_dim']))
    d.y = torch.ones((d.batch_size, d.input_dim))

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


def test_forward_pass_of_nsf_log_prob(data_setup_nsf_small):
    """
    Test the forward pass of flow model for log_prob.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net)

    loss_0 = model(d.y, d.x, d.z)
    with pytest.raises(ValueError):
        model(d.y, d.z, d.x)
    with pytest.raises(RuntimeError):
        model(d.y, d.x, d.x)
    # TODO: check output


# TODO: Tests for sampling and backward passes
