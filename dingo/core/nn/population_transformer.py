from types import SimpleNamespace

import torch
import torch.nn as nn

from dingo.core.nn.enets import DenseResidualNet


class PopulationTransformer(nn.Module):
    def __init__(self, tokenizer, transformer_encoder, final_net=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer_encoder = transformer_encoder

        self.final_net = final_net

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.tokenizer(src)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mask again before averaging. Note that averaging includes the zero elements
        # in the sequence.
        if src_key_padding_mask is not None:
            x = x * ~src_key_padding_mask.unsqueeze(-1)
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

    encoder = PopulationTransformer(tokenizer, transformer, final_net)
    return encoder
