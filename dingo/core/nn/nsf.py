"""
Implementation of the neural spline flow (NSF). Most of this code is adapted
from the uci.py example from https://github.com/bayesiains/nsf.
"""

import torch
import torch.nn as nn
import glasflow.nflows as nflows  # nflows not maintained, so use this maintained fork
from glasflow.nflows import distributions, flows, transforms
import glasflow.nflows.nn.nets as nflows_nets
from dingo.core.nn.resnet import DenseResidualNet
from dingo.core.utils import torchutils


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
    layer_norm: bool = False,
    conditioner_type: str = "glasflow_residual",
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
    :param layer_norm: bool = False
        whether to use layer normalization in the conditioner network
        (conditioner_type "dense_residual" only)
    :param conditioner_type: str = "glasflow_residual"
        conditioner network of the rq-coupling transform. "glasflow_residual"
        (glasflow's ResidualNet, context concatenated to the input once) or
        "dense_residual" (dingo's DenseResidualNet, context injected into every
        residual block via a gated linear unit, optional layer_norm). The two are
        architecturally different — checkpoints are not interchangeable — so this
        is an explicit type, not a flag.
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
    if layer_norm and conditioner_type != "dense_residual":
        raise ValueError(
            "layer_norm requires conditioner_type 'dense_residual' (glasflow's "
            "ResidualNet only supports batch norm)."
        )

    if base_transform_type == "rq-coupling":
        if param_dim == 1:
            mask = torch.tensor([1], dtype=torch.uint8)
        else:
            mask = nflows.utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)
            )

        if conditioner_type == "glasflow_residual":
            transform_net_create_fn = (
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
            )
        elif conditioner_type == "dense_residual":
            transform_net_create_fn = (
                lambda in_features, out_features: DenseResidualNet(
                    input_dim=in_features,
                    output_dim=out_features,
                    hidden_dims=(hidden_dim,) * num_transform_blocks,
                    activation=activation_fn,
                    context_features=context_dim,
                    dropout=dropout_probability,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                )
            )
        else:
            raise ValueError(
                f"Unknown conditioner_type '{conditioner_type}'; expected "
                f"'glasflow_residual' or 'dense_residual'."
            )

        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )

    elif base_transform_type == "rq-autoregressive":
        if conditioner_type != "glasflow_residual":
            raise ValueError(
                "rq-autoregressive only supports conditioner_type "
                "'glasflow_residual'."
            )
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
    This class wraps the neural spline flow, and routes named context tensors into
    the embedding network. It is required for multiple reasons. (i) The embedding
    network can consume several context tensors (declared in context_keys), which is
    not supported by the nflows package. (ii) Parallelization across multiple GPUs
    requires a forward method, but the relevant flow method for training is log_prob.
    """

    def __init__(
        self,
        flow: flows.base.Flow,
        embedding_net: nn.Module = None,
        context_keys: tuple = ("waveform",),
    ):
        """
        :param flow: flows.base.Flow
        :param embedding_net: nn.Module
        :param context_keys: tuple
            Keys of the context dict that the embedding network consumes, in the
            order of its forward arguments.
        """
        super(FlowWrapper, self).__init__()
        self.embedding_net = embedding_net
        self.flow = flow
        self.context_keys = tuple(context_keys)

    def _embed_context(self, context: dict):
        """Select the tensors in context_keys (in order) and embed them."""
        missing = [k for k in self.context_keys if k not in context]
        if missing:
            raise ValueError(
                f"Context is missing keys {missing}: expected {self.context_keys}, "
                f"got {sorted(context)}."
            )
        x = [context[k] for k in self.context_keys]
        if self.embedding_net is not None:
            return self.embedding_net(*x)
        if len(x) != 1:
            raise ValueError("Multiple context tensors require an embedding network.")
        return x[0]

    def log_prob(self, y, context: dict = None):
        if context is None:
            return self.flow.log_prob(y)
        return self.flow.log_prob(y, self._embed_context(context))

    def sample(self, context: dict = None, num_samples: int = 1):
        if context is None:
            return self.flow.sample(num_samples)
        return self.flow.sample(num_samples, self._embed_context(context))

    def sample_and_log_prob(self, context: dict = None, num_samples: int = 1):
        if context is None:
            return self.flow.sample_and_log_prob(num_samples)
        return self.flow.sample_and_log_prob(num_samples, self._embed_context(context))

    def forward(self, y, context: dict = None):
        return self.log_prob(y, context)


def create_nsf_model(
    input_dim: int,
    context_dim: int,
    num_flow_steps: int,
    base_transform_kwargs: dict,
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
    :return: Flow
        the NSF (posterior model)
    """
    distribution = distributions.StandardNormal((input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, context_dim, base_transform_kwargs
    )
    flow = flows.Flow(transform, distribution)

    return flow


if __name__ == "__main__":
    pass
