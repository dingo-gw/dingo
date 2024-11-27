from types import SimpleNamespace

import torch
import torch.nn as nn

from dingo.core.nn.enets import DenseResidualNet


class PopulationTransformer(nn.Module):
    def __init__(self, tokenizer, transformer_encoder, final_net=None, use_variance=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer_encoder = transformer_encoder
        self.final_net = final_net
        self.use_variance = use_variance

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize parameters of transformer encoder explicitly due to Issue
        https://github.com/pytorch/pytorch/issues/72253.
        The parameters of the transformer encoder are initialized with xavier uniform.
        """

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.tokenizer(src)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Average over non-masked components.
        if src_key_padding_mask is not None:
            denominator = torch.sum(~src_key_padding_mask, -1, keepdim=True)
            
            if self.use_variance:
                x1 = torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2) / denominator
                # Compute variance
                x2 = torch.sum((x ** 2) * ~src_key_padding_mask.unsqueeze(-1), dim=-2) / denominator
                # Concatenate mean and variance
                x = torch.cat([x1, x2], axis=-1)
            else:
                x = torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2) / denominator

        else:
            if self.use_variance:
                x1 = torch.mean(x, dim=-2)
                # Compute variance
                x2 = torch.mean(x ** 2, dim=-2)
                # Concatenate mean and variance
                x = torch.cat([x1, x2], axis=-1)
            else:
                x = torch.mean(x, dim=-2)

        if self.final_net is not None:
            x = self.final_net(x)
        return x


def create_population_transformer(config):
    """Builder function for a transformer based multi-event encoder."""
    # autocomplete config
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    config.tokenizer["input_dim"] = config.d_dingo_encoding
    config.tokenizer["output_dim"] = config.transformer["d_model"]

    use_variance = config.transformer.get("use_variance", False)

    if(use_variance):
        config.final_net["input_dim"] = 2*config.transformer["d_model"]
    else:
        config.final_net["input_dim"] = config.transformer["d_model"]

    # build individual modules
    tokenizer = DenseResidualNet(**config.tokenizer)
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=config.transformer["d_model"],
        dim_feedforward=config.transformer.get("dim_feedforward", 2048),
        nhead=config.transformer["nhead"],
        dropout=config.transformer.get("dropout", 0.1),
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(
        transformer_layer, num_layers=config.transformer["num_layers"]
    )
    final_net = DenseResidualNet(**config.final_net)

    encoder = PopulationTransformer(tokenizer, transformer, final_net, use_variance=use_variance)
    return encoder
