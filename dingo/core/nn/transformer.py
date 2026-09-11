from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dingo.core.nn.resnet import DenseResidualNet, LinearLayer
from dingo.core.utils import torchutils

# Disable the fused TransformerEncoderLayer fast path (_transformer_encoder_layer_fwd).
# It raises a CUDA illegal memory access on sm_100 (B200) for fp32 eval-mode forwards
# (torch 2.13.0+cu130; the same code runs on sm_90/H100). The fast path is only
# reachable in eval mode without autocast, and its nested-tensor optimization is off
# anyway for norm_first=True and for token masks that are not left-aligned, so
# disabling it costs nothing here.
torch.backends.mha.set_fastpath_enabled(False)


class TokenEmbedding(nn.Module):
    """
    Maps each token's raw features to a d_model-dimensional embedding via a shared
    DenseResidualNet, conditioned on the token's position.

    A position is a vector of position_continuous_dim continuous features followed by
    len(position_category_sizes) categorical indices, the i-th taking values in
    [0, position_category_sizes[i]). The continuous features are used as given and the
    categorical ones are one-hot encoded; their concatenation is the GLU context of
    the residual blocks. (GW tokens: lower and upper frequency + detector index.)

    Methods
    -------
    forward:
        Obtain the token embedding for a Tensor of shape
        [..., num_tokens, num_features], conditioned on position.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Callable,
        position_continuous_dim: int,
        position_category_sizes: Sequence[int],
        dropout: float = 0.0,
        norm: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        input_dim : int
            number of features per token (the last dimension of the tokenized data)
        hidden_dims : List[int]
            dimensions of hidden layers for the underlying DenseResidualNet
        output_dim : int
            output dimension of the token embedding (typically d_model)
        activation : Callable
            activation function for the DenseResidualNet
        position_continuous_dim : int
            number of continuous position features (the leading position columns)
        position_category_sizes : Sequence[int]
            number of categories of each categorical position feature (the trailing
            position columns, in order); each is one-hot encoded
        dropout : float
            dropout rate for the DenseResidualNet
        norm : str or None
            normalization used in the DenseResidualNet: "LayerNorm" or None.
            "BatchNorm" is not supported (raises): the residual net runs on
            [..., num_tokens, features] and nn.BatchNorm1d normalizes over axis 1,
            the token axis
        """
        super().__init__()
        if norm == "BatchNorm":
            raise ValueError(
                "BatchNorm is not supported in TokenEmbedding: nn.BatchNorm1d treats "
                "axis 1 of the [..., num_tokens, features] input as the channel axis, "
                "i.e. it would normalize per token position. Use LayerNorm instead."
            )
        self.num_features = input_dim
        self.position_continuous_dim = position_continuous_dim
        self.position_category_sizes = list(position_category_sizes)
        self.tokenizer_net = DenseResidualNet(
            input_dim=self.num_features,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            context_features=position_continuous_dim
            + sum(self.position_category_sizes),
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            shape [..., num_tokens, num_features]
        position : Tensor
            shape [..., num_tokens, position_dim] with position_dim =
            position_continuous_dim + len(position_category_sizes): the continuous
            features followed by the categorical indices

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
        position_dim = self.position_continuous_dim + len(self.position_category_sizes)
        if position.shape[-1] != position_dim:
            raise ValueError(
                f"Expected positions with {position_dim} features "
                f"({self.position_continuous_dim} continuous + "
                f"{len(self.position_category_sizes)} categorical), got "
                f"{position.shape[-1]}."
            )
        context = [position[..., : self.position_continuous_dim]]
        for i, size in enumerate(self.position_category_sizes):
            index = position[..., self.position_continuous_dim + i].long()
            context.append(F.one_hot(index, size).to(position.dtype))
        return self.tokenizer_net(x=x, context=torch.cat(context, dim=-1))


class TransformerModel(nn.Module):
    """
    Transformer encoder used as an embedding network for the normalizing flow. Each
    token is embedded via a TokenEmbedding (conditioned on position), then
    processed by a standard TransformerEncoder. The resulting sequence of token
    embeddings is pooled (CLS token or average) into a single vector, optionally
    followed by a final network.
    """

    def __init__(
        self,
        tokenizer: TokenEmbedding,
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
        tokenizer : TokenEmbedding
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

        # Attribute name kept for state-dict compatibility with saved networks.
        self.tokenizer = tokenizer
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
                # A sample with every token masked averages to zero instead of NaN.
                denominator = torch.sum(
                    ~src_key_padding_mask, dim=-1, keepdim=True
                ).clamp(min=1)
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


def create_transformer_enet(
    tokenizer_kwargs: dict,
    transformer_kwargs: dict,
    pooling: str = "cls",
    final_net_kwargs: Optional[dict] = None,
) -> TransformerModel:
    """
    Builder function for a transformer embedding network for tokenized 1D data with
    multiple blocks (detectors) and channels.

    Parameters
    ----------
    tokenizer_kwargs : dict
        settings for the TokenEmbedding. Must contain input_dim,
        position_continuous_dim and position_category_sizes (set from the data by
        autocomplete_model_kwargs, not hardcoded in a settings file); activation is
        given as a str and resolved to a Callable here; output_dim is set
        automatically to transformer_kwargs["d_model"].
    transformer_kwargs : dict
        settings for the TransformerModel: d_model, dim_feedforward, nhead,
        num_layers, dropout, norm_first.
    pooling : str
        one of ["average", "cls"]
    final_net_kwargs : Optional[dict]
        settings for the network applied after pooling. Must contain output_dim. If
        it also contains hidden_dims, a DenseResidualNet is built and activation is
        required (dropout and norm are then read from this dict as
        well, analogous to tokenizer_kwargs). Otherwise, a single linear layer is
        used, followed by activation if one is given. If final_net_kwargs is None,
        no final_net is used and the pooled d_model-dim vector is returned directly.

    Returns
    -------
    TransformerModel
    """
    tokenizer_kwargs = dict(tokenizer_kwargs)
    tokenizer_kwargs["activation"] = torchutils.get_activation_function_from_string(
        tokenizer_kwargs["activation"]
    )
    tokenizer = TokenEmbedding(
        output_dim=transformer_kwargs["d_model"],
        **tokenizer_kwargs,
    )

    final_net = None
    if final_net_kwargs is not None:
        final_net_kwargs = dict(final_net_kwargs)
        output_dim = final_net_kwargs.pop("output_dim")
        if final_net_kwargs.get("activation") is not None:
            final_net_kwargs["activation"] = (
                torchutils.get_activation_function_from_string(
                    final_net_kwargs["activation"]
                )
            )
        elif "hidden_dims" in final_net_kwargs:
            raise ValueError("final_net_kwargs with hidden_dims requires activation.")
        if "hidden_dims" in final_net_kwargs:
            final_net_kwargs["hidden_dims"] = tuple(final_net_kwargs["hidden_dims"])
            final_net = DenseResidualNet(
                input_dim=transformer_kwargs["d_model"],
                output_dim=output_dim,
                **final_net_kwargs,
            )
        else:
            final_net = LinearLayer(
                input_dim=transformer_kwargs["d_model"],
                output_dim=output_dim,
                **final_net_kwargs,
            )

    return TransformerModel(
        tokenizer=tokenizer,
        pooling=pooling,
        final_net=final_net,
        **transformer_kwargs,
    )
