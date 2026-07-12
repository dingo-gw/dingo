import pytest
import types
import torch
import torch.optim as optim
from dingo.core.nn.nsf import create_nsf_model, FlowWrapper
from dingo.core.nn.enets import (
    ConcatContextMerger,
    DenseSVDEmbedding,
    MLPContextMerger,
    create_enet_with_projection_layer_and_dense_resnet,
)
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
    d.context = {"waveform": d.x, "context_parameters": d.z}

    # build d.yy, which depends on input d.zz
    d.xx = torch.cat((d.x, d.x))
    d.yy = torch.cat((d.y, -d.y))
    d.zz = torch.cat((d.z, -d.z))
    d.contextcontext = {"waveform": d.xx, "context_parameters": d.zz}

    return d


# context_keys for embedding networks built with added_context=True.
CONTEXT_KEYS = ("waveform", "context_parameters")


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
    model = FlowWrapper(flow, embedding_net, CONTEXT_KEYS)

    samples = model.sample(d.context)
    # model.sample(num_samples=1) adds an extra dimension that needs to be squeezed.
    samples = samples.squeeze(1)

    assert samples.shape == d.y.shape, "Unexpected shape of samples."
    assert torch.all(samples > -10) and torch.all(samples < 10), (
        "Unexpected samples encountered. Network initialization or "
        "normalization seems broken."
    )

    # missing context key
    with pytest.raises(ValueError):
        model.sample({"waveform": d.x})
    # wrongly-shaped context tensor
    with pytest.raises(RuntimeError):
        model.sample({"waveform": d.x, "context_parameters": d.x.flatten(start_dim=1)})


def test_forward_pass_for_log_prob_of_nsf(data_setup_nsf_small):
    """
    Test the forward pass of flow model for log_prob.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net, CONTEXT_KEYS)

    loss = -model(d.y, d.context)
    assert list(loss.shape) == [d.batch_size], "Unexpected output shape."
    assert torch.all(loss > 0) and torch.all(loss < 40), (
        "Unexpected log prob encountered. Network initialization or "
        "normalization seems broken."
    )

    # missing context key
    with pytest.raises(ValueError):
        model(d.y, {"waveform": d.x})
    # wrongly-shaped context tensor
    with pytest.raises(RuntimeError):
        model(d.y, {"waveform": d.x, "context_parameters": d.x.flatten(start_dim=1)})


def test_backward_pass_for_log_prob_of_nsf(data_setup_nsf_small):
    """
    Test the backward pass of flow model for log_prob. This function also
    checks that conditional sampling improves during training.
    """

    d = data_setup_nsf_small

    embedding_net = d.embedding_net_builder(**d.embedding_net_kwargs)
    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, embedding_net, CONTEXT_KEYS)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Simple train loop. The learned parameters yy are strongly correlated
    # with the context information d.zz. We check that (i) the loss and (ii)
    # the posterior samples improve during training.
    losses = []
    for idx in range(40):
        yy = d.yy + 0.02 * torch.rand_like(d.yy)
        xx = torch.rand_like(d.xx)
        loss = -torch.mean(model(yy, {"waveform": xx, "context_parameters": d.zz}))
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    samples_n = torch.mean(model.sample(d.contextcontext, num_samples=100), axis=1)

    assert losses[-1] < losses[0], "Loss did not improve in training."
    assert (
        torch.mean((torch.abs(samples_n - d.yy) > 0.8).float()) < 0.3
    ), "Training may not have worked. Check manually that sampling improves."


def test_registered_embedding_with_merger(data_setup_nsf_small):
    """
    Test the registered dense_svd embedding and the concat context merger: contract
    attributes, dimension inference, and end-to-end use inside a FlowWrapper.
    """

    d = data_setup_nsf_small
    kwargs = {
        k: v
        for k, v in d.embedding_net_kwargs.items()
        if k not in ("added_context", "V_rb_list", "input_dims")
    }
    num_context_parameters = d.context_dim - kwargs["output_dim"]

    completed = DenseSVDEmbedding.complete_settings(
        kwargs, {"waveform": d.x[0], "context_parameters": d.z[0]}
    )
    assert completed["input_dims"] == list(d.embedding_net_kwargs["input_dims"])

    embedding_net = DenseSVDEmbedding(**completed)
    assert embedding_net.input_keys == ("waveform",)
    assert embedding_net.output_dim == kwargs["output_dim"]

    merged = ConcatContextMerger(embedding_net, num_context_parameters)
    assert merged.input_keys == CONTEXT_KEYS
    assert merged.output_dim == d.context_dim
    assert (
        ConcatContextMerger.merged_output_dim(
            embedding_net.output_dim, num_context_parameters
        )
        == d.context_dim
    )

    # State-dict layout matches the historic builder (old checkpoints).
    legacy = create_enet_with_projection_layer_and_dense_resnet(
        **d.embedding_net_kwargs
    )
    assert list(merged.state_dict()) == list(legacy.state_dict())

    flow = d.nde_builder(**d.nde_kwargs)
    model = FlowWrapper(flow, merged, merged.input_keys)

    loss = -model(d.y, d.context)
    assert list(loss.shape) == [d.batch_size], "Unexpected output shape."
    assert torch.all(loss > 0) and torch.all(loss < 40), (
        "Unexpected log prob encountered. Network initialization or "
        "normalization seems broken."
    )


def test_mlp_context_merger(data_setup_nsf_small):
    """
    Test the mlp context merger (ported ContextMergerMLP): the context parameters
    are mixed in through a learned MLP, so the merged output dimension does not
    grow with the number of context parameters.
    """

    d = data_setup_nsf_small
    kwargs = {
        k: v
        for k, v in d.embedding_net_kwargs.items()
        if k not in ("added_context", "V_rb_list")
    }
    num_context_parameters = d.z.shape[1]
    embedding_net = DenseSVDEmbedding(**kwargs)

    merged = MLPContextMerger(
        embedding_net, num_context_parameters, hidden_dims=[16, 16]
    )
    assert merged.input_keys == CONTEXT_KEYS
    # By default, the merged output dimension equals the embedding output dim.
    assert merged.output_dim == embedding_net.output_dim
    assert (
        MLPContextMerger.merged_output_dim(
            embedding_net.output_dim,
            num_context_parameters=num_context_parameters,
            hidden_dims=[16, 16],
        )
        == embedding_net.output_dim
    )

    out = merged(d.x, d.z)
    assert out.shape == (d.batch_size, embedding_net.output_dim)

    # An explicit output_dim overrides the default.
    merged_wide = MLPContextMerger(
        embedding_net, num_context_parameters, hidden_dims=[16], output_dim=12
    )
    assert merged_wide(d.x, d.z).shape == (d.batch_size, 12)
    assert (
        MLPContextMerger.merged_output_dim(
            embedding_net.output_dim,
            num_context_parameters=num_context_parameters,
            hidden_dims=[16],
            output_dim=12,
        )
        == 12
    )


def test_dense_residual_conditioner(data_setup_nsf_small):
    """The rq-coupling conditioner network is an explicit type: dense_residual
    (GLU context, optional layer_norm) builds a working flow; layer_norm without
    it, or unknown types, are errors."""
    from dingo.core.nn.resnet import DenseResidualNet as DingoDenseResidualNet

    d = data_setup_nsf_small
    kwargs = dict(d.nde_kwargs)
    kwargs["base_transform_kwargs"] = {
        **d.base_transform_kwargs,
        "batch_norm": False,
        "conditioner_type": "dense_residual",
        "layer_norm": True,
    }
    flow = create_nsf_model(**kwargs)
    # The conditioner networks inside the coupling transforms are dingo's
    # DenseResidualNet, one per flow step.
    conditioners = [m for m in flow.modules() if isinstance(m, DingoDenseResidualNet)]
    assert len(conditioners) == d.num_flow_steps

    context_vector = torch.rand(d.batch_size, d.context_dim)
    log_prob = flow.log_prob(d.y, context_vector)
    assert log_prob.shape == (d.batch_size,)
    assert torch.isfinite(log_prob).all()

    # layer_norm requires the dense_residual conditioner.
    with pytest.raises(ValueError, match="layer_norm"):
        create_nsf_model(
            **{
                **d.nde_kwargs,
                "base_transform_kwargs": {
                    **d.base_transform_kwargs,
                    "layer_norm": True,
                },
            }
        )
    # Unknown conditioner types are an error.
    with pytest.raises(ValueError, match="conditioner_type"):
        create_nsf_model(
            **{
                **d.nde_kwargs,
                "base_transform_kwargs": {
                    **d.base_transform_kwargs,
                    "conditioner_type": "foo",
                },
            }
        )
