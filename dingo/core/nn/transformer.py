"""Transformer embedding network for tokenized strain data (DINGO-T1)."""

from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dingo.core.nn.resnet import DenseResidualNet, LinearLayer
from dingo.core.registry import EMBEDDING_NETS
from dingo.core.utils import torchutils


class Tokenizer(nn.Module):
    """
    Maps each token's raw features to a d_model-dimensional embedding via a shared
    DenseResidualNet, conditioned on the token's position (f_min, f_max, detector).

    Methods
    -------
    forward:
        Obtain the token embedding for a Tensor of shape
        [..., num_tokens, num_features], conditioned on position.
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dims: List[int],
        output_dim: int,
        activation: Callable,
        num_blocks: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        """
        Parameters
        ----------
        input_dims : List[int]
            [num_tokens, num_features], i.e., the shape of the tokenized waveform,
            omitting batch dimensions. Only num_features (the last entry) is used.
        hidden_dims : List[int]
            dimensions of hidden layers for the underlying DenseResidualNet
        output_dim : int
            output dimension of the token embedding (typically d_model)
        activation : Callable
            activation function for the DenseResidualNet
        num_blocks : int
            number of blocks (detectors, in the GW use case); determines the size of
            the one-hot detector encoding used as part of the conditioning context
        dropout : float
            dropout rate for the DenseResidualNet
        batch_norm : bool
            whether to use batch normalization in the DenseResidualNet
        layer_norm : bool
            whether to use layer normalization in the DenseResidualNet
        """
        super().__init__()
        if len(input_dims) != 2:
            raise ValueError(
                f"Invalid shape in Tokenizer, expected len(input_dims) == 2, got "
                f"{input_dims}."
            )
        self.num_features = input_dims[-1]
        self.num_blocks = num_blocks
        self.tokenizer_net = DenseResidualNet(
            input_dim=self.num_features,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            context_features=2 + num_blocks,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )

    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            shape [..., num_tokens, num_features]
        position : Tensor
            shape [..., num_tokens, 3], last dim = [f_min, f_max, detector_index]

        Returns
        -------
        Tensor
            shape [..., num_tokens, output_dim]
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Invalid shape for token embedding layer. "
                f"Expected last dimension to be {self.num_features}, got "
                f"{x.shape[-1]}."
            )
        detector_per_token = position[..., 2]
        detector_one_hot = torch.eye(self.num_blocks, device=position.device)[
            detector_per_token.long()
        ]
        context = torch.cat((position[..., :2], detector_one_hot), dim=-1)
        return self.tokenizer_net(x=x, context=context)


class TransformerModel(nn.Module):
    """
    Transformer encoder used as an embedding network for the normalizing flow. Each
    token is embedded via a conditional Tokenizer (conditioned on position), then
    processed by a standard TransformerEncoder. The resulting sequence of token
    embeddings is pooled (CLS token or average) into a single vector, optionally
    followed by a final network.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        pooling: str = "cls",
        final_net: Optional[nn.Module] = None,
    ):
        """
        Parameters
        ----------
        tokenizer : Tokenizer
            Maps raw per-token features (conditioned on position) to d_model-dim
            token embeddings.
        d_model : int
            embedding size of the transformer
        dim_feedforward : int
            number of hidden dimensions in the feedforward networks of the
            transformer encoder layers
        nhead : int
            number of transformer attention heads
        num_layers : int
            number of transformer encoder layers
        dropout : float
            dropout probability in the transformer encoder layers
        norm_first : bool
            if True, layer normalization is applied before the attention and
            feedforward operations in each encoder layer, otherwise after
        pooling : str
            one of ["average", "cls"]; how to pool the sequence of token embeddings
            into a single vector
        final_net : Optional[nn.Module]
            network applied to the pooled output, e.g., to project it to the
            context dimension expected by the normalizing flow. If None, the pooled
            output is returned directly.
        """
        super().__init__()
        if pooling not in ("average", "cls"):
            raise ValueError(
                f"Invalid pooling operation {pooling}, expected one of "
                f"['average', 'cls']."
            )

        self.tokenizer = tokenizer
        self.d_model = d_model
        self.pooling = pooling
        self.final_net = final_net

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        if self.pooling == "cls":
            self.class_token = nn.Parameter(torch.randn((1, 1, d_model)))

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize parameters of the transformer encoder explicitly, due to
        https://github.com/pytorch/pytorch/issues/72253. Parameters are initialized
        with xavier uniform.
        """
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: Tensor,
        position: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            shape [batch_size, num_tokens, num_features]
        position : Tensor
            shape [batch_size, num_tokens, 3], last dim = [f_min, f_max, detector]
        src_key_padding_mask : Optional[Tensor]
            shape [batch_size, num_tokens]; PyTorch transformer convention, True =
            masked out (not allowed to attend)

        Returns
        -------
        Tensor
            shape [batch_size, output_dim of final_net if final_net else d_model]
        """
        x = self.tokenizer(x=x, position=position)

        if self.pooling == "cls":
            batch_size = x.shape[0]
            x = torch.cat((self.class_token.expand(batch_size, -1, -1), x), dim=1)
            if src_key_padding_mask is not None:
                mask_cls_token = torch.zeros(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=src_key_padding_mask.device,
                )
                src_key_padding_mask = torch.cat(
                    (mask_cls_token, src_key_padding_mask), dim=1
                )

        x = self.transformer_encoder(src=x, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == "average":
            if src_key_padding_mask is not None:
                denominator = torch.sum(~src_key_padding_mask, dim=-1, keepdim=True)
                x = (
                    torch.sum(x * (~src_key_padding_mask).unsqueeze(-1), dim=-2)
                    / denominator
                )
            else:
                x = torch.mean(x, dim=-2)
        else:  # pooling == "cls"
            x = x[..., 0, :]

        if self.final_net is not None:
            x = self.final_net(x)

        return x


@EMBEDDING_NETS.register("transformer")
class TransformerEmbedding(TransformerModel):
    """
    TransformerModel as a registered embedding network (see the contract in
    dingo.core.nn.enets): consumes the tokenized batch entries produced by
    StrainTokenization, and builds tokenizer / final net from settings dicts.
    """

    input_keys = ("waveform", "position", "drop_token_mask")

    def __init__(
        self,
        tokenizer_kwargs: dict,
        transformer_kwargs: dict,
        output_dim: int,
        pooling: str = "cls",
        final_net_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        tokenizer_kwargs : dict
            Settings for the Tokenizer: hidden_dims, activation (str), and
            optionally dropout, batch_norm, layer_norm. input_dims and num_blocks
            are inferred from a sample batch by complete_settings; the tokenizer
            output_dim is transformer_kwargs["d_model"].
        transformer_kwargs : dict
            Settings for the transformer encoder: d_model, dim_feedforward, nhead,
            num_layers, and optionally dropout, norm_first.
        output_dim : int
            Dimension of the embedded context: the output_dim of final_net_kwargs
            if given, else d_model. Inferred by complete_settings; not a user
            setting.
        pooling : str
            one of ["average", "cls"]
        final_net_kwargs : Optional[dict]
            Settings for the network applied after pooling. Must contain
            output_dim and activation (str). With hidden_dims, a DenseResidualNet
            is built (dropout, batch_norm, layer_norm are then read as well);
            otherwise a LinearLayer. If None, the pooled d_model-dim vector is
            returned directly.
        """
        tokenizer_kwargs = dict(tokenizer_kwargs)
        tokenizer_kwargs["activation"] = torchutils.get_activation_function_from_string(
            tokenizer_kwargs["activation"]
        )
        tokenizer = Tokenizer(
            output_dim=transformer_kwargs["d_model"],
            **tokenizer_kwargs,
        )

        final_net = None
        if final_net_kwargs is not None:
            final_net_kwargs = dict(final_net_kwargs)
            final_net_output_dim = final_net_kwargs.pop("output_dim")
            final_net_kwargs["activation"] = (
                torchutils.get_activation_function_from_string(
                    final_net_kwargs["activation"]
                )
            )
            if "hidden_dims" in final_net_kwargs:
                final_net_kwargs["hidden_dims"] = tuple(final_net_kwargs["hidden_dims"])
                final_net = DenseResidualNet(
                    input_dim=transformer_kwargs["d_model"],
                    output_dim=final_net_output_dim,
                    **final_net_kwargs,
                )
            else:
                final_net = LinearLayer(
                    input_dim=transformer_kwargs["d_model"],
                    output_dim=final_net_output_dim,
                    **final_net_kwargs,
                )
        else:
            final_net_output_dim = transformer_kwargs["d_model"]
        if output_dim != final_net_output_dim:
            raise ValueError(
                f"Inconsistent settings: output_dim is {output_dim}, but the "
                f"network produces {final_net_output_dim} "
                f"(final_net output_dim, or d_model without a final net)."
            )

        super().__init__(
            tokenizer=tokenizer,
            pooling=pooling,
            final_net=final_net,
            **transformer_kwargs,
        )
        self.output_dim = output_dim

    @classmethod
    def complete_settings(cls, settings: dict, sample_batch: dict) -> dict:
        """Infer the tokenizer input dims and number of blocks (detectors) plus the
        embedding output_dim from a sample batch; return completed settings."""
        tokenizer_kwargs = dict(settings["tokenizer_kwargs"])
        for key in ("input_dims", "num_blocks"):
            if key in tokenizer_kwargs:
                raise ValueError(
                    f"'{key}' is derived from the data and must not be specified "
                    f"in the tokenizer settings."
                )
        if "output_dim" in settings:
            raise ValueError(
                "'output_dim' is derived from the network settings and must not "
                "be specified."
            )
        # sample_batch["waveform"] is the tokenized strain, [num_tokens,
        # num_features]; column 2 of position holds integer detector indices
        # 0..num_blocks-1.
        tokenizer_kwargs["input_dims"] = list(sample_batch["waveform"].shape)
        tokenizer_kwargs["num_blocks"] = int(sample_batch["position"][:, 2].max()) + 1

        final_net_kwargs = settings.get("final_net_kwargs")
        if final_net_kwargs is not None:
            output_dim = final_net_kwargs["output_dim"]
        else:
            output_dim = settings["transformer_kwargs"]["d_model"]
        return {
            **settings,
            "tokenizer_kwargs": tokenizer_kwargs,
            "output_dim": output_dim,
        }
