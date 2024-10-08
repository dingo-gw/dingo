"""
Implementation of the neural spline flow (NSF). Most of this code is adapted
from the uci.py example from https://github.com/bayesiains/nsf.
"""

import copy

import torch
import torch.nn as nn
import glasflow.nflows as nflows  # nflows not maintained, so use this maintained fork
from glasflow.nflows import distributions, flows, transforms
import glasflow.nflows.nn.nets as nflows_nets
from dingo.core.utils import torchutils
from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet
from typing import Union, Callable, Tuple


def create_linear_transform(param_dim: int):
    """
    Create the composite linear transform PLU.

    :param param_dim: int
        dimension of the parameter space
    :return: nde.Transform
        the linear transform PLU
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_base_transform(
    i: int,
    param_dim: int,
    context_dim: int = None,
    hidden_dim: int = 512,
    num_transform_blocks: int = 2,
    activation: str = "relu",
    dropout_probability: float = 0.0,
    batch_norm: bool = False,
    num_bins: int = 8,
    tail_bound: float = 1.0,
    apply_unconditional_transform: bool = False,
    base_transform_type: str = "rq-coupling",
):
    """
    Build a base NSF transform of y, conditioned on x.

    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
    in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).

    Code is adapted from the uci.py example from
    https://github.com/bayesiains/nsf.

    A coupling flow fixes half the components of y, and applies a transform
    to the remaining components, conditioned on the fixed components. This is
    a restricted form of an autoregressive transform, with a single split into
    fixed/transformed components.

    The transform here is a neural spline flow, where the flow is parametrized
    by a residual neural network that depends on y_fixed and x. The residual
    network consists of a sequence of two-layer fully-connected blocks.

    :param i: int
        index of transform in sequence
    :param param_dim: int
        dimensionality of y
    :param context_dim: int = None
        dimensionality of x
    :param hidden_dim: int = 512
        number of hidden units per layer
    :param num_transform_blocks: int = 2
        number of transform blocks comprising the transform
    :param activation: str = 'relu'
        activation function
    :param dropout_probability: float = 0.0
        dropout probability for regularization
    :param batch_norm: bool = False
        whether to use batch normalization
    :param num_bins: int = 8
        number of bins for the spline
    :param tail_bound: float = 1.
    :param apply_unconditional_transform: bool = False
        whether to apply an unconditional transform to fixed components
    :param base_transform_type: str = 'rq-coupling'
        type of base transform, one of {rq-coupling, rq-autoregressive}

    :return: Transform
        the NSF transform
    """

    activation_fn = torchutils.get_activation_function_from_string(activation)

    if base_transform_type == "rq-coupling":
        if param_dim == 1:
            mask = torch.tensor([1], dtype=torch.uint8)
        else:
            mask = nflows.utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)
            )
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=(
                lambda in_features, out_features: nflows_nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=num_transform_blocks,
                    activation=activation_fn,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm,
                )
            ),
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )

    elif base_transform_type == "rq-autoregressive":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=param_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=activation_fn,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm,
        )

    else:
        raise ValueError


def create_transform(
    num_flow_steps: int, param_dim: int, context_dim: int, base_transform_kwargs: dict
):
    """
    Build a sequence of NSF transforms, which maps parameters y into the
    base distribution u (noise). Transforms are conditioned on context data x.

    Note that the forward map is f^{-1}(y, x).

    Each step in the sequence consists of
        * A linear transform of y, which in particular permutes components
        * A NSF transform of y, conditioned on x.
    There is one final linear transform at the end.

    :param num_flow_steps: int,
        number of transforms in sequence
    :param param_dim: int,
        dimensionality of parameter space (y)
    :param context_dim: int,
        dimensionality of context (x)
    :param base_transform_kwargs: int
        hyperparameters for NSF step
    :return: Transform
        the NSF transform sequence
    """

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(param_dim),
                    create_base_transform(
                        i, param_dim, context_dim=context_dim, **base_transform_kwargs
                    ),
                ]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform(param_dim)]
    )

    return transform


class FlowWrapper(nn.Module):
    """
    This class wraps the neural spline flow. It is required for multiple
    reasons. (i) some embedding networks take tuples as input, which is not
    supported by the nflows package. (ii) paralellization across multiple
    GPUs requires a forward method, but the relevant flow method for training
    is log_prob.
    """

    def __init__(self, flow: flows.base.Flow, embedding_net: nn.Module = None):
        """

        :param flow: flows.base.Flow
        :param embedding_net: nn.Module
        """
        super(FlowWrapper, self).__init__()
        self.embedding_net = embedding_net
        self.flow = flow

    def log_prob(self, y, *x):
        if self.embedding_net is not None:
            x = torchutils.forward_pass_with_unpacked_tuple(self.embedding_net, x)
        if len(x) > 0:
            return self.flow.log_prob(y, x)
        else:
            # if there is no context
            return self.flow.log_prob(y)

    def sample(self, *x, num_samples=1):
        if self.embedding_net is not None:
            x = torchutils.forward_pass_with_unpacked_tuple(self.embedding_net, x)
        if len(x) > 0:
            return self.flow.sample(num_samples, x)
        else:
            # if there is no context, omit the context argument
            return self.flow.sample(num_samples)

    def sample_and_log_prob(self, *x, num_samples=1):
        if self.embedding_net is not None:
            x = torchutils.forward_pass_with_unpacked_tuple(self.embedding_net, x)
        if len(x) > 0:
            return self.flow.sample_and_log_prob(num_samples, x)
        else:
            # if there is no context, omit the context argument
            return self.flow.sample_and_log_prob(num_samples)

    def forward(self, y, *x):
        if len(x) > 0:
            return self.log_prob(y, *x)
        else:
            # if there is no context, omit the context argument
            return self.log_prob(y)


def create_nsf_model(
    input_dim: int,
    context_dim: int,
    num_flow_steps: int,
    base_transform_kwargs: dict,
    embedding_net_builder: Union[Callable, str] = None,
    embedding_net_kwargs: dict = None,
):
    """
    Build NSF model. This models the posterior distribution p(y|x).

    The model consists of
        * a base distribution (StandardNormal, dim(y))
        * a sequence of transforms, each conditioned on x

    :param input_dim: int,
        dimensionality of y
    :param context_dim: int,
        dimensionality of the (embedded) context
    :param num_flow_steps: int,
        number of sequential transforms
    :param base_transform_kwargs: dict,
        hyperparameters for transform steps
    :param embedding_net_builder: Callable=None,
        build function for embedding network TODO
    :param embedding_net_kwargs: dict=None,
        hyperparameters for embedding network
    :return: Flow
        the NSF (posterior model)
    """

    if embedding_net_builder is not None:
        embedding_net = embedding_net_builder(**embedding_net_kwargs)
    else:
        embedding_net = None

    # str(embedding_net_builder).split(' ')[1]

    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, context_dim, base_transform_kwargs
    )
    flow = flows.Flow(transform, distribution, embedding_net)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "num_flow_steps": num_flow_steps,
        "context_dim": context_dim,
        "base_transform_kwargs": base_transform_kwargs,
        "embedding_net_kwargs": embedding_net_kwargs,
    }

    return flow


def create_nsf_wrapped(**kwargs):
    """
    Wraps the NSF model in a FlowWrapper. This is required for parallel
    training, and wraps the log_prob method as a forward method.
    """
    flow = create_nsf_model(**kwargs)
    return FlowWrapper(flow)


def create_nsf_with_rb_projection_embedding_net(
    nsf_kwargs: dict,
    embedding_net_kwargs: dict,
    initial_weights: dict = None,
):
    """Builds a neural spline flow with an embedding network that consists of a
    reduced basis projection followed by a residual network. Optionally initializes the
    embedding network weights.

    Parameters
    ----------
    nsf_kwargs : dict
        kwargs for neural spline flow
    embedding_net_kwargs : dict
        kwargs for emebedding network
    initial_weights : dict
        Dictionary containing the initial weights for the SVD projection. This should
        have one key 'V_rb_list', with value a list of SVD V matrices (one for each
        detector).

    Returns
    -------
    nn.Module
        Neural spline flow model
    """
    # We copy the embedding_net_kwargs to allow an insert of V_rb_list without
    # affecting the original embedding_net_kwargs. This is because we don't want to
    # save the embedding_net_kwargs with the huge V_rb_list included. This is a bit of
    # a hack; improve setting of initial weights later.

    embedding_net_kwargs = copy.deepcopy(embedding_net_kwargs)
    if initial_weights is not None:
        embedding_net_kwargs["V_rb_list"] = initial_weights["V_rb_list"]
    elif "V_rb_list" not in embedding_net_kwargs:
        embedding_net_kwargs["V_rb_list"] = None

    embedding_net = create_enet_with_projection_layer_and_dense_resnet(
        **embedding_net_kwargs
    )
    flow = create_nsf_model(**nsf_kwargs)
    model = FlowWrapper(flow, embedding_net)
    return model


def autocomplete_model_kwargs_nsf(model_kwargs, data_sample):
    """
    Autocomplete the model kwargs from train_settings and data_sample from
    the dataloader:
    (*) set input dimension of embedding net to shape of data_sample[1]
    (*) set dimension of nsf parameter space to len(data_sample[0])
    (*) set added_context flag of embedding net if required for gnpe proxies
    (*) set context dim of nsf to output dim of embedding net + gnpe proxy dim

    :param train_settings: dict
        train settings as loaded from .yaml file
    :param data_sample: list
        Sample from dataloader (e.g., wfd[0]) used for autocomplection.
        Should be of format [parameters, GW data, gnpe_proxies], where the
        last element is only there is gnpe proxies are required.
    :return: model_kwargs: dict
        updated, autocompleted model_kwargs
    """
    # set input dims from ifo_list and domain information
    model_kwargs["embedding_net_kwargs"]["input_dims"] = list(data_sample[1].shape)
    # set dimension of parameter space of nsf
    model_kwargs["nsf_kwargs"]["input_dim"] = len(data_sample[0])
    # set added_context flag of embedding net if gnpe proxies are required
    # set context dim of nsf to output dim of embedding net + gnpe proxy dim
    try:
        gnpe_proxy_dim = len(data_sample[2])
        model_kwargs["embedding_net_kwargs"]["added_context"] = True
        model_kwargs["nsf_kwargs"]["context_dim"] = (
            model_kwargs["embedding_net_kwargs"]["output_dim"] + gnpe_proxy_dim
        )
    except IndexError:
        model_kwargs["embedding_net_kwargs"]["added_context"] = False
        model_kwargs["nsf_kwargs"]["context_dim"] = model_kwargs[
            "embedding_net_kwargs"
        ]["output_dim"]
    # return model_kwargs


if __name__ == "__main__":
    pass
