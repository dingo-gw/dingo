from typing import Callable, Tuple

import torch
from torch import nn
from torch.nn import functional as F, init


class MyResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs.
    This is taken from nflows, but modified to allow for LayerNorm instead of
    BatchNorm1D."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        use_layer_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        if use_batch_norm and use_layer_norm:
            raise ValueError(
                "Residual block should not use both batch norm and layer " "norm."
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

    def forward(self, inputs, context=None):
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
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class DenseResidualNet(nn.Module):
    """
    A nn.Module consisting of a sequence of dense residual blocks. This is
    used to embed high dimensional input to a compressed output. Linear
    resizing layers are used for resizing the input and output to match the
    first and last hidden dimension, respectively.

    Module specs
    --------
        input dimension:    (batch_size, input_dim)
        output dimension:   (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple,
        activation: Callable = F.elu,
        context_features: int = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
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
        activation: callable
            activation function used in residual blocks
        context_features: int=None
            Number of additional context features, which are provided to the residual
            blocks via gated linear units. If None, no additional context expected.
        dropout: float=0.0
            dropout probability for residual blocks used for reqularization
        batch_norm: bool=True
            flag that specifies whether to use batch normalization
        layer_norm: bool=False
            flag that specifies whether to use layer normalization
        """

        super(DenseResidualNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

        # This attribute is required by nflows.
        if all([d == self.hidden_dims[0] for d in self.hidden_dims]):
            self.hidden_features = self.hidden_dims[0]

        if context_features is not None:
            self.initial_layer = nn.Linear(input_dim + context_features, hidden_dims[0])
        else:
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

    def forward(self, x, context=None):
        if context is None:
            x = self.initial_layer(x)
        else:
            # Should dim=-1 below to allow for seq dimension?
            x = self.initial_layer(torch.cat((x, context), dim=1))
        for block, resize_layer in zip(self.blocks, self.resize_layers):
            x = block(x, context=context)
            x = resize_layer(x)
        return x
