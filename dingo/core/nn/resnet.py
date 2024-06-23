import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs.

    Adapted from glasflow.nflows.nn.nets.resnet, but with support for layernorm.
    """

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

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("Can't use batch norm and layer norm simultaneously.")

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
