from types import SimpleNamespace

import torch
import torch.nn as nn

from dingo.core.nn.enets import DenseResidualNet


class PopulationTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        transformer_encoder,
        final_net=None,
        use_moments=None,
        pooling="average",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer_encoder = transformer_encoder
        self.final_net = final_net
        self.use_moments = use_moments
        self.pooling = pooling

        if self.pooling == "cls":
            self.class_token = nn.Parameter(
                torch.randn((1, 1, self.tokenizer.output_dim))
            )

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

        if self.pooling == "cls":
            # Prepend the class token.
            batch_size = x.shape[0]
            x = torch.cat((self.class_token.expand(batch_size, -1, -1), x), dim=1)
            if src_key_padding_mask is not None:
                # Extend the mask.
                cls_mask = torch.zeros(
                    (batch_size, 1),
                    dtype=src_key_padding_mask.dtype,
                    device=src_key_padding_mask.device,
                )
                src_key_padding_mask = torch.cat(
                    [cls_mask, src_key_padding_mask], dim=1
                )

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == "average":
            # Average over non-masked components.
            if src_key_padding_mask is not None:
                denominator = torch.sum(~src_key_padding_mask, -1, keepdim=True)

                if self.use_moments is not None:
                    moments = []
                    if "mean" in self.use_moments:
                        mean = (
                            torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2)
                            / denominator
                        )
                        moments.append(mean)
                    if "variance" in self.use_moments:
                        variance = (
                            torch.sum(
                                (x**2) * ~src_key_padding_mask.unsqueeze(-1), dim=-2
                            )
                            / denominator
                        )
                        moments.append(variance)
                    if "skewness" in self.use_moments:
                        skewness = (
                            torch.sum(
                                (x**3) * ~src_key_padding_mask.unsqueeze(-1), dim=-2
                            )
                            / denominator
                        )
                        moments.append(skewness)
                    if "kurtosis" in self.use_moments:
                        kurtosis = (
                            torch.sum(
                                (x**4) * ~src_key_padding_mask.unsqueeze(-1), dim=-2
                            )
                            / denominator
                        )
                        moments.append(kurtosis)

                    x = torch.cat(moments, dim=-1)
                else:
                    x = (
                        torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2)
                        / denominator
                    )

            else:
                # TODO: write this up properly. STEPHEN: I commented this, since it was
                #  not clear what was happening.
                # src_key_padding_mask = (src == 0).all(dim=-1)
                #
                # return self.forward(src, src_key_padding_mask)

                raise NotImplementedError(
                    "PopulationTransformer does not support "
                    "average pooling without padding mask."
                )
                # if self.use_moments is not None:
                #     moments = []
                #     if 'mean' in self.use_moments:
                #         mean = torch.mean(x, dim=-2)
                #         moments.append(mean)
                #     if 'variance' in self.use_moments:
                #         variance = torch.mean((x ** 2), dim=-2)
                #         moments.append(variance)
                #     if 'skewness' in self.use_moments:
                #         skewness = torch.mean((x ** 3), dim=-2)
                #         moments.append(skewness)
                #     if 'kurtosis' in self.use_moments:
                #         kurtosis = torch.mean((x ** 4), dim=-2)
                #         moments.append(kurtosis)
                #
                #     x = torch.cat(moments, axis=-1)
                # else:
                #     x = torch.mean(x, dim=-2)

        elif self.pooling == "cls":
            # Take the first token.
            x = x[:, 0, :]
        else:
            raise NotImplementedError(
                "Pooling operation not implemented. Should not be here."
            )

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

    # TODO, tempoary fix for old versions
    use_variance = config.transformer.get("use_variance", False)
    if use_variance:
        use_moments = ["mean", "variance"]
    else:
        use_moments = ["mean"]

    use_moments = config.transformer.get("use_moments", use_moments)

    if use_moments is not None:
        n = len(use_moments)
        config.final_net["input_dim"] = n * config.transformer["d_model"]
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

    encoder = PopulationTransformer(
        tokenizer,
        transformer,
        final_net,
        use_moments=use_moments,
        pooling=config.transformer.get("pooling", "average"),
    )
    return encoder
