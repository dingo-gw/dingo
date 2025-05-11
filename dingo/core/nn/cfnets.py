import copy

import numpy as np
import torch
import torch.nn as nn

from dingo.core.utils import torchutils
from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet

from dingo.core.nn.enets import DenseResidualNet


class ContinuousFlow(nn.Module):
    """
    A continuous normalizing flow network. It defines a time-dependent vector field on
    the parameter space (score or flow), which optionally depends on additional context
    information.

    v = v(f(t, theta), g(context))

    This class combines the network v for the continuous flow itself, as well as embedding
    networks f, g, for the context and parameters, respectively.

    The parameters and context can optionally be provided as gated linear unit (GLU)
    context to the main network, rather than as the main input to the network. For a
    DenseResidualNet, this context is input repeatedly via GLUs, for each residual block.
    """

    def __init__(
        self,
        continuous_flow_net: nn.Module,
        context_embedding_net: nn.Module = torch.nn.Identity(),
        theta_embedding_net: nn.Module = torch.nn.Identity(),
        context_with_glu: bool = False,
        theta_with_glu: bool = False,
    ):
        """
        Parameters
        ----------
        continuous_flow_net: nn.Module
            Main network for the continuous flow.
        context_embedding_net: nn.Module = torch.nn.Identity()
            Embedding network for the context information (e.g., observed data).
        theta_embedding_net: nn.Module = torch.nn.Identity()
            Embedding network for the parameters.
        context_with_glu: bool = False
            Whether to provide context as GLU or main input to the continuous_flow_net.
        theta_with_glu: bool = False
            Whether to provide theta (and t) as GLU or main input to the
            continuous_flow_net.
        """
        super(ContinuousFlow, self).__init__()
        self.continuous_flow_net = continuous_flow_net
        self.context_embedding_net = context_embedding_net
        self.theta_embedding_net = theta_embedding_net
        self.theta_with_glu = theta_with_glu
        self.context_with_glu = context_with_glu

        self._use_cache = None
        self._cached_context = None
        self._cached_context_embedding = None

    @property
    def use_cache(self):
        # unless set explicitly, use_cache is True in eval mode and False in train mode
        if self._use_cache is not None:
            return self._use_cache
        else:
            return not self.training

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value

    def _update_cached_context(self, *context: torch.Tensor):
        """
        Update the cache for *context. This sets new values for self._cached_context and
        self._cached_context_embedding if self._cached_context != context.
        """
        try:
            # This may fail when batch size of context and _cached_context is different
            # (but both > 1).
            if (
                self._cached_context is not None
                and len(self._cached_context) == len(context)
                and all([(x == y).all() for x, y in zip(self._cached_context, context)])
            ):
                return
        except RuntimeError:
            pass
        # if all tensors in batch are the same: do forward pass with batch_size 1
        if all([(x == x[:1]).all() for x in context]):
            self._cached_context = tuple(x[:1] for x in context)
            self._cached_context_embedding = self.context_embedding_net(
                *self._cached_context
            ).detach()

        else:
            self._cached_context = context
            self._cached_context_embedding = self.context_embedding_net(
                *self._cached_context
            ).detach()

    def _get_cached_context_embedding(self, batch_size):
        if self._cached_context_embedding.size(0) == 1:
            return self._cached_context_embedding.repeat(
                batch_size,
                *[1 for _ in range(len(self._cached_context_embedding.shape) - 1)],
            )
        return self._cached_context_embedding

    def forward(self, t, theta, *context):
        # embed theta (self.embedding_net_theta might just be identity)
        t_and_theta_embedding = torch.cat((t.unsqueeze(1), theta), dim=1)
        t_and_theta_embedding = self.theta_embedding_net(t_and_theta_embedding)
        # for unconditional forward pass
        if len(context) == 0:
            assert not self.theta_with_glu
            return self.continuous_flow_net(t_and_theta_embedding)

        # embed context (self.context_embedding_net might just be identity)
        if not self.use_cache:
            context_embedding = self.context_embedding_net(*context)

        else:
            self._update_cached_context(*context)
            context_embedding = self._get_cached_context_embedding(
                batch_size=len(context[0])
            )

        if len(t_and_theta_embedding.shape) != 2 or len(context_embedding.shape) != 2:
            raise NotImplementedError()

        # a = context_embedding and b = t_and_theta_embedding now need to be provided
        # to the continuous flow network, which predicts a vector field as a function
        # of a and b. The flow network has two entry points: the normal input to the
        # feedforward network (first argument in forward pass) and via a glu between
        # the residual blocks (second argument in forward pass, optional). The flags
        # self.theta_with_glu and self.context_with_glu specify whether we use the
        # first entrypoint (= False) or the second (= True).
        if self.context_with_glu and self.theta_with_glu:
            main_input = torch.Tensor([])
            glu_context = torch.cat((context_embedding, t_and_theta_embedding), dim=1)
        elif not self.context_with_glu and not self.theta_with_glu:
            main_input = torch.cat((context_embedding, t_and_theta_embedding), dim=1)
            glu_context = None
        elif self.context_with_glu:
            main_input = t_and_theta_embedding
            glu_context = context_embedding
        else:  # if self.theta_with_glu:
            main_input = context_embedding
            glu_context = t_and_theta_embedding

        if glu_context is None:
            return self.continuous_flow_net(main_input)
        else:
            return self.continuous_flow_net(main_input, glu_context)


def create_cf(
    posterior_kwargs: dict, embedding_kwargs: dict = None, initial_weights: dict = None
):
    """
    Build a continuous flow based on settings dictionaries.

    Parameters
    ----------
    posterior_kwargs: dict
        Settings for the flow. This includes the settings for the parameter embedding.
    embedding_kwargs: dict
        Settings for the context embedding network.
    initial_weights: dict
        Initial weights for the embedding network (of SVD projection type).

    Returns
    -------
    nn.Module
        Neural network for the continuous flow.
    """
    theta_dim = posterior_kwargs["input_dim"]
    context_dim = posterior_kwargs["context_dim"]

    # get embeddings modules for context
    if embedding_kwargs is not None:
        context_embedding_kwargs = copy.deepcopy(embedding_kwargs)
        if initial_weights is not None:
            context_embedding_kwargs["V_rb_list"] = initial_weights["V_rb_list"]
        elif "V_rb_list" not in context_embedding_kwargs:
            context_embedding_kwargs["V_rb_list"] = None

        context_embedding = create_enet_with_projection_layer_and_dense_resnet(
            **context_embedding_kwargs
        )
    else:
        context_embedding = torch.nn.Identity()

    # get embeddings modules for theta (which is actually cat(t, theta))
    if "theta_embedding_kwargs" in posterior_kwargs:
        theta_embedding = get_theta_embedding_net(
            posterior_kwargs["theta_embedding_kwargs"],
            input_dim=theta_dim + 1,
        )
    else:
        theta_embedding = torch.nn.Identity()

    # get output dimensions of embedded context and theta
    theta_with_glu = posterior_kwargs.get("theta_with_glu", False)
    context_with_glu = posterior_kwargs.get("context_with_glu", False)
    embedded_theta_dim = theta_embedding(torch.zeros(10, theta_dim + 1)).shape[1]

    glu_dim = theta_with_glu * embedded_theta_dim + context_with_glu * context_dim
    input_dim = embedded_theta_dim + context_dim - glu_dim
    if glu_dim == 0:
        glu_dim = None

    activation_fn = torchutils.get_activation_function_from_string(
        posterior_kwargs["activation"]
    )
    continuous_flow_net = DenseResidualNet(
        input_dim=input_dim,
        output_dim=theta_dim,
        hidden_dims=posterior_kwargs["hidden_dims"],
        activation=activation_fn,
        dropout=posterior_kwargs["dropout"],
        batch_norm=posterior_kwargs["batch_norm"],
        context_features=glu_dim,
    )

    model = ContinuousFlow(
        continuous_flow_net,
        context_embedding,
        theta_embedding,
        theta_with_glu=posterior_kwargs.get("theta_with_glu", False),
        context_with_glu=posterior_kwargs.get("context_with_glu", False),
    )
    return model


def get_theta_embedding_net(embedding_kwargs: dict, input_dim):
    if "encoding" in embedding_kwargs:
        input_dim = get_dim_positional_embedding(
            embedding_kwargs["encoding"], input_dim
        )
        positional_encoding = PositionalEncoding(
            nr_frequencies=embedding_kwargs.get("frequencies", 0),
            encode_all=embedding_kwargs.get("encode_all"),
        )
    else:
        positional_encoding = torch.nn.Identity()

    if "embedding_net" in embedding_kwargs:
        activation_fn = torchutils.get_activation_function_from_string(
            embedding_kwargs["embedding_net"]["activation"]
        )
        embedding_net = DenseResidualNet(
            input_dim=input_dim,
            output_dim=embedding_kwargs["embedding_net"]["output_dim"],
            hidden_dims=embedding_kwargs["embedding_net"]["hidden_dims"],
            activation=activation_fn,
            dropout=embedding_kwargs["embedding_net"].get("dropout", 0.0),
            batch_norm=embedding_kwargs["embedding_net"].get("batch_norm", True),
        )
    else:
        embedding_net = torch.nn.Identity()

    return torch.nn.Sequential(positional_encoding, embedding_net)


def get_dim_positional_embedding(encoding: dict, input_dim: int):
    if encoding.get("encode_all"):
        return (1 + 2 * encoding["frequencies"]) * input_dim
    return 2 * encoding["frequencies"] + input_dim

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as commonly used in transformer architectures.
    
    Positional encoding introduces a way to inject information about the order of 
    the input data (e.g., sequence positions) into a neural network that otherwise 
    lacks a sense of position due to its permutation-invariant nature. This class 
    computes sinusoidal encodings based on the position of each element in the input 
    and concatenates them with the original input features.

    Attributes
    ----------
    frequencies : torch.Tensor
        A tensor containing the frequencies used to calculate the sinusoidal components.
        The frequencies are powers of 2, scaled by the base frequency.
    encode_all : bool
        Determines whether the positional encoding is applied to all features of the input
        or only the first feature (e.g., time component).
    base_freq : float
        The base frequency used to scale the sinusoidal components, defaulting to `2 * pi`.

    Parameters
    ----------
    nr_frequencies : int
        The number of sinusoidal frequencies to compute. This determines the dimensionality
        of the positional encoding for each input feature.
    encode_all : bool, optional (default=True)
        If True, the positional encoding is computed for all features in the input. 
        Otherwise, it is computed only for the first feature (e.g., the time dimension).
    base_freq : float, optional (default=2 * np.pi)
        The base frequency used for sinusoidal encoding.

    Methods
    -------
    forward(t_theta)
        Computes the positional encoding for the input tensor `t_theta` and concatenates 
        it with the original input features.
        - If `encode_all` is True, the positional encoding is computed for all features.
        - If `encode_all` is False, the positional encoding is applied only to the first
          feature, such as time, while other features remain unchanged.
    """
    def __init__(self, nr_frequencies, encode_all=True, base_freq=2 * np.pi):
        super(PositionalEncoding, self).__init__()
        frequencies = base_freq * torch.pow(
            2 * torch.ones(nr_frequencies), torch.arange(0, nr_frequencies)
        ).view(1, 1, nr_frequencies)
        self.register_buffer("frequencies", frequencies)
        self.encode_all = encode_all

    def forward(self, t_theta):
        """
        Computes and concatenates positional encodings with the input tensor.

        Parameters
        ----------
        t_theta : torch.Tensor
            Input tensor of shape (batch_size, input_dim), where `input_dim` is the
            dimensionality of the input features.

        Returns
        -------
        torch.Tensor
            A tensor containing the input features concatenated with the positional
            encodings. The output shape will be:
            - (batch_size, input_dim + 2 * nr_frequencies) if `encode_all` is True.
            - (batch_size, input_dim + 2 * nr_frequencies) if `encode_all` is False,
              but positional encodings are computed only for the first input feature.
        """
        batch_size = t_theta.size(0)
        if self.encode_all:
            x = t_theta.view(batch_size, -1, 1) * self.frequencies
        else:
            x = t_theta[:, 0:1].view(batch_size, 1, 1) * self.frequencies
        cos_enc, sin_enc = torch.cos(x).view(batch_size, -1), torch.sin(x).view(
            batch_size, -1
        )
        return torch.cat((t_theta, cos_enc, sin_enc), dim=1)
