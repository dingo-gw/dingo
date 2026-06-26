"""Dense residual network supporting layer normalization and multi-dimensional
(e.g., token-batched) inputs, used by the transformer tokenizer."""

from typing import Callable, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F, init


class MyResidualBlock(nn.Module):
    """
    A general-purpose residual block, supporting batch norm or layer norm.

    Context features are injected via a gated linear unit. The GLU is applied along
    the last dimension, so this supports both [batch, features] and
    [batch, tokens, features] inputs.
    """

    def __init__(
        self,
        features: int,
        context_features: Optional[int] = None,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
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
        use_batch_norm : bool
            whether to use batch normalization
        use_layer_norm : bool
            whether to use layer normalization
        zero_initialization : bool
            whether to initialize the final linear layer with small weights
        """
        super().__init__()
        self.activation = activation

        if use_batch_norm and use_layer_norm:
            raise ValueError(
                "Residual block should not use both batch norm and layer norm."
            )
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if use_layer_norm:
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

    def forward(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        if self.use_layer_norm:
            temps = self.layer_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        if self.use_layer_norm:
            temps = self.layer_norm_layers[1](temps)
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

    1. **Context conditioning**: glasflow's ResidualNet concatenates the context vector
       with the input once at the start of the network. Here, context is injected into
       every residual block via a gated linear unit (GLU), giving the network repeated
       access to the conditioning information at each layer.

    2. **Normalization and input shape**: glasflow's ResidualNet only supports batch
       normalization and 2D [batch, features] inputs. This implementation adds layer
       normalization and supports inputs with an arbitrary number of leading batch
       dimensions (e.g., [batch, tokens, features]), as needed by the transformer
       tokenizer.

    Because of difference (1), the two classes are not interchangeable: a checkpoint
    trained with glasflow's ResidualNet cannot be loaded into DenseResidualNet.

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
        context_features: Optional[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
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
        activation : Callable
            activation function used in residual blocks
        context_features : Optional[int]
            number of additional context features, which are provided to the
            residual blocks via gated linear units; if None, no context expected
        dropout : float
            dropout probability for residual blocks, used for regularization
        batch_norm : bool
            whether to use batch normalization
        layer_norm : bool
            whether to use layer normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

        self.initial_layer = nn.Linear(self.input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                MyResidualBlock(
                    features=self.hidden_dims[n],
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout,
                    use_batch_norm=batch_norm,
                    use_layer_norm=layer_norm,
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
    """A single linear layer followed by an activation function."""

    def __init__(self, input_dim: int, output_dim: int, activation: Callable):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))
