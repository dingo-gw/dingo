import copy

import numpy as np
import torch
import torch.nn as nn

from dingo.core.utils import torchutils
from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet
from typing import Union, Callable

from dingo.core.nn.enets import DenseResidualNet


# Autocomplete model kwargs necessary?


# TODO make this inherent from an abstract wrapper?
class ContinuousFlowModel(nn.Module):
    """
    This class wraps the continuous flow models. It combines an embedding net and a continuous flow model that
     learns a vector field (score or flow).

    Methods
    -------

    set_embedding:
        infers and caches the embedding of the given context
    get_embedding:
        infers the embedding of given context, returns cached context if no context is provided
    forward:
        calls model, evaluates first the embedding and then the continuous flow
    """

    def __init__(
        self,
        continuous_flow: nn.Module,
        context_embedding_net: nn.Module = torch.nn.Identity(),
        theta_embedding_net: nn.Module = torch.nn.Identity(),
        context_with_glu: bool = False,
        theta_with_glu: bool = False,
    ):
        """
        TODO
        :param  continuous_flow: nn.Module
        :param embedding_net: nn.Module
        """
        super(ContinuousFlowModel, self).__init__()
        self.continuous_flow = continuous_flow
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

    def _update_cached_context(self, *context):
        """
        Update the cache for *context. This sets new values for self._cached_context and
        self._cached_context_embedding if self._cached_context != context.
        """
        # if self._cached_context is None:
        #     print("TODO: Slightly rewritten context caching to adopt to varying batch_size, pls check")
        try:
            # This may fail when batch size of context and _cached_context is different
            # (but both > 1).
            if (
                self._cached_context is not None
                and len(self._cached_context) == len(context)
                and all([(x == y).all() for x, y in zip(self._cached_context, context)])
            ):
                return
        except:
            pass
        # if all tensors in batch are the same: do forward pass with batch_size 1
        if all([(x == x[:1]).all() for x in context]):
            self._cached_context = [x[:1] for x in context]
            self._cached_context_embedding = torchutils.forward_pass_with_unpacked_tuple(
                self.context_embedding_net, *self._cached_context
            ).detach()
        else:
            self._cached_context = context
            self._cached_context_embedding = (
                torchutils.forward_pass_with_unpacked_tuple(
                    self.context_embedding_net, *self._cached_context
                )
            ).detach()

    def _get_cached_context_embedding(self, batch_size):
        if self._cached_context_embedding.size(0) == 1:
            return self._cached_context_embedding.repeat(
                batch_size, *[1 for _ in range(len(self._cached_context_embedding.shape) - 1)]
            )
        return self._cached_context_embedding

    def forward(self, t, theta, *context):
        # embed theta (self.embedding_net_theta might just be identity)
        t_and_theta_embedding = torch.cat((t.unsqueeze(1), theta), dim=1)
        t_and_theta_embedding = self.theta_embedding_net(t_and_theta_embedding)
        # for unconditional forward pass
        if len(context) == 0:
            assert not self.theta_with_glu
            return self.continuous_flow(t_and_theta_embedding)

        # embed context (self.context_embedding_net might just be identity)
        if not self.use_cache:
            context_embedding = torchutils.forward_pass_with_unpacked_tuple(
                self.context_embedding_net, *context
            )
        else:
            self._update_cached_context(*context)
            context_embedding = self._get_cached_context_embedding(batch_size=len(context[0]))

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
            input = torch.Tensor([])
            glu_context = torch.cat((context_embedding, t_and_theta_embedding), dim=1)
        elif not self.context_with_glu and not self.theta_with_glu:
            input = torch.cat((context_embedding, t_and_theta_embedding), dim=1)
            glu_context = None
        elif self.context_with_glu:
            input = t_and_theta_embedding
            glu_context = context_embedding
        else:  # if self.theta_with_glu:
            input = context_embedding
            glu_context = t_and_theta_embedding

        if glu_context is None:
            return self.continuous_flow(input)
        else:
            return self.continuous_flow(input, glu_context)


def create_cf_model(
    posterior_kwargs: dict, embedding_kwargs: dict = None, initial_weights: dict = None
):
    """
    TODO: re-name 'embedding_kwargs' to 'context_embedding_kwargs'.
    Build CF model. This models the posterior distribution p(y|x).

    The model consists of
        * an embedding
        * a continuous flow model

    :param input_dim: int,
        dimensionality of theta
    :param context_dim: int,
        dimensionality of the (embedded) context
    :param posterior_kwargs: dict,
        posterior_kwargs characterizing the cf-net hyperparamters
    :param embedding_net_builder: Callable=None,
        build function for embedding network TODO
    :param embedding_kwargs: dict=None,
        hyperparameters for embedding network
    :return: ContinuousFlowModel
        the cf (posterior model)
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
    continuous_flow = DenseResidualNet(
        input_dim=input_dim,
        output_dim=theta_dim,
        hidden_dims=posterior_kwargs["hidden_dims"],
        activation=activation_fn,
        dropout=posterior_kwargs["dropout"],
        batch_norm=posterior_kwargs["batch_norm"],
        context_features=glu_dim,
    )

    model = ContinuousFlowModel(
        continuous_flow,
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
    def __init__(self, nr_frequencies, encode_all=True, base_freq=2 * np.pi):
        super(PositionalEncoding, self).__init__()
        frequencies = base_freq * torch.pow(
            2 * torch.ones(nr_frequencies), torch.arange(0, nr_frequencies)
        ).view(1, 1, nr_frequencies)
        self.register_buffer("frequencies", frequencies)
        self.encode_all = encode_all

    def forward(self, t_theta):
        batch_size = t_theta.size(0)
        if self.encode_all:
            x = t_theta.view(batch_size, -1, 1) * self.frequencies
        else:
            x = t_theta[:, 0:1].view(batch_size, 1, 1) * self.frequencies
        cos_enc, sin_enc = torch.cos(x).view(batch_size, -1), torch.sin(x).view(
            batch_size, -1
        )
        return torch.cat((t_theta, cos_enc, sin_enc), dim=1)
