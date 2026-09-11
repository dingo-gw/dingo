"""Dense residual networks with GLU context gating, a selectable normalization
layer, and support for multi-dimensional (e.g., token-batched) inputs. Used by the
transformer token embedding, as the NSF coupling conditioner when norm="LayerNorm",
and by the continuous-flow (FMPE) networks."""

from typing import Callable, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F, init

NORM_OPTIONS = ("BatchNorm", "LayerNorm", None)


def check_norm_option(norm: Optional[str]) -> None:
    if norm not in NORM_OPTIONS:
        raise ValueError(
            f"norm must be 'BatchNorm', 'LayerNorm' or None, got {norm!r}."
        )


class MyResidualBlock(nn.Module):
    """
    A general-purpose residual block.

    This is taken from nflows, but modified to allow for LayerNorm instead of
    BatchNorm1d, and to apply the context GLU along the last dimension so that
    token-batched ``[batch, tokens, features]`` inputs are gated per token
    (LayerNorm or no normalization only; BatchNorm1d treats dim 1 as the channel
    axis). The parameter names (``batch_norm_layers``, ``linear_layers``,
    ...) match those of the nflows ``ResidualBlock`` so that state dicts of
    networks trained with BatchNorm remain loadable; LayerNorm parameters are
    stored under ``layer_norm_layers`.
    """

    def __init__(
        self,
        features: int,
        context_features: Optional[int] = None,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        norm: Optional[str] = None,
        zero_initialization: bool = True,
    ):
        """
        Parameters
        ----------
        features : int
            dimension of the residual block input and output
        context_features : Optional[int]
            number of context features injected via a gated linear unit; if None,
            no context is expected
        activation : Callable
            activation function used between linear layers
        dropout_probability : float
            dropout probability applied for regularization
        norm : str or None
            normalization used in the block: "BatchNorm", "LayerNorm" or None
        zero_initialization : bool
            whether to initialize the final linear layer with small weights
        """
        super().__init__()
        check_norm_option(norm)
        self.activation = activation
        self.norm = norm

        if norm == "BatchNorm":
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        elif norm == "LayerNorm":
            self.layer_norm_layers = nn.ModuleList(
                [nn.LayerNorm(features) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def _normalize(self, x: Tensor, index: int) -> Tensor:
        if self.norm == "BatchNorm":
            return self.batch_norm_layers[index](x)
        if self.norm == "LayerNorm":
            return self.layer_norm_layers[index](x)
        return x

    def forward(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        temps = self._normalize(inputs, 0)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        temps = self._normalize(temps, 1)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=-1), dim=-1
            )
        return inputs + temps


class DenseResidualNet(nn.Module):
    """
    A nn.Module consisting of a sequence of dense residual blocks. This is
    used to embed high dimensional input to a compressed output. Linear
    resizing layers are used for resizing the input and output to match the
    first and last hidden dimension, respectively.

    Compared to glasflow.nflows.nn.nets.ResidualNet, which the residual blocks here are
    based on, this implementation differs in two important ways:

    1. **Context conditioning**: glasflow's ResidualNet both concatenates the context
       to the input of the initial layer and injects it into every residual block via
       a gated linear unit (GLU). This implementation keeps the per-block GLU
       injection but omits the input concatenation.

    2. **Normalization and input shape**: glasflow's ResidualNet only supports batch
       normalization and 2D [batch, features] inputs. This implementation adds layer
       normalization and supports inputs with an arbitrary number of leading batch
       dimensions (e.g., [batch, tokens, features]), as needed by the transformer
       token embedding.

    Because of difference (1) -- and differently named layers -- the two classes are
    not state-dict compatible: a checkpoint trained with glasflow's ResidualNet
    cannot be loaded into DenseResidualNet (the residual blocks themselves are
    compatible).

    Module specs
    --------
        input dimension:    (..., input_dim)
        output dimension:   (..., output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple,
        activation: Callable = F.elu,
        dropout: float = 0.0,
        norm: Optional[str] = "BatchNorm",
        context_features: int = None,
    ):
        """
        Parameters
        ----------
        input_dim : int
            dimension of the input to this module
        output_dim : int
            output dimension of this module
        hidden_dims : tuple
            tuple with dimensions of hidden layers of this module
        activation: callable
            activation function used in residual blocks
        dropout: float
            dropout probability for residual blocks used for reqularization
        norm: str or None
            normalization used in the residual blocks: "BatchNorm", "LayerNorm"
            or None
        context_features: int
            Number of additional context features, which are provided to the residual
            blocks via gated linear units. If None, no additional context expected.
        """

        super().__init__()
        check_norm_option(norm)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

        # Read by nflows' piecewise coupling transforms (duck-typed, hasattr): when
        # this attribute exists, the conditioner's spline widths/heights are divided
        # by sqrt(hidden_features). Removing it changes the flow, and networks
        # trained with it (e.g. the published Dingo-T1 network) give wrong
        # posteriors without it. Requires a single hidden width.
        if all(d == self.hidden_dims[0] for d in self.hidden_dims):
            self.hidden_features = self.hidden_dims[0]

        self.initial_layer = nn.Linear(self.input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                MyResidualBlock(
                    features=self.hidden_dims[n],
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout,
                    norm=norm,
                )
                for n in range(self.num_res_blocks)
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                (
                    nn.Linear(self.hidden_dims[n - 1], self.hidden_dims[n])
                    if self.hidden_dims[n - 1] != self.hidden_dims[n]
                    else nn.Identity()
                )
                for n in range(1, self.num_res_blocks)
            ]
            + [nn.Linear(self.hidden_dims[-1], self.output_dim)]
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = self.initial_layer(x)
        for block, resize_layer in zip(self.blocks, self.resize_layers):
            x = block(x, context=context)
            x = resize_layer(x)
        return x


class LinearLayer(nn.Module):
    """A single linear layer, followed by an activation function if one is given."""

    def __init__(
        self, input_dim: int, output_dim: int, activation: Optional[Callable] = None
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return x if self.activation is None else self.activation(x)
