import math
from typing import Callable, List, Union

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dingo.core.nn.resnet import DenseResidualNet, MLP
from dingo.core.utils import torchutils


class Tokenizer(nn.Module):
    """
    A nn.Module that maps each frequency fragment of length num_bins_per_token and width num_channels
    into an embedding vector of fixed d_model via (the same or individual) linear layers.

    Methods
    -------
    forward:
        obtain the token embedding for a Tensor of shape [batch_size, num_tokens, num_features]
         = [batch_size, num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
        where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dims: Union[List, int],
        output_dim: int,
        activation: Callable,
        dropout: float = 0.0,
        batch_norm: bool = True,
        layer_norm: bool = False,
        individual_token_embedding: bool = False,
    ):
        """
        Parameters
        --------
        input_dims: List[int]
            containing [num_tokens, num_features]=[num_tokens_per_block * num_blocks, num_channels*num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
        hidden_dims: Optional[Tuple, int]
            if type List: dimensions of hidden layers for DenseResidualNet
            if type int: number of nodes in linear layer
        output_dim: int
            output dimension
        activation: str
            activation function for DenseResidualNet
        dropout: float
            dropout rate for DenseResidualNet
        batch_norm: bool
            whether to use batch normalization for DenseResidualNet
        layer_norm: bool
            whether to use layer normalization for DenseResidualNet
        individual_token_embedding: bool
            whether to use an individual linear layer for each token
        """
        super(Tokenizer, self).__init__()

        assert (
            len(input_dims) == 2
        ), f"Invalid shape in Tokenizer, expected len(input_dims) == 2, got {input_dims})."
        self.num_tokens, self.num_features = input_dims
        self.individual_token_embedding = individual_token_embedding
        if self.individual_token_embedding:
            if isinstance(hidden_dims, list):
                self.stack_tokenizer_nets = nn.ModuleList(
                    [
                        DenseResidualNet(
                            input_dim=self.num_features,
                            output_dim=output_dim,
                            hidden_dims=tuple(hidden_dims),
                            activation=activation,
                            dropout=dropout,
                            batch_norm=batch_norm,
                            layer_norm=layer_norm,
                        )
                        for _ in range(self.num_tokens)
                    ]
                )
            elif isinstance(hidden_dims, int):
                self.stack_embedding_networks = nn.ModuleList(
                    [
                        MLP(
                            input_size=self.num_features,
                            hidden_size=hidden_dims,
                            output_size=output_dim,
                            activation_fn=activation,
                        )
                        for _ in range(self.num_tokens)
                    ]
                )
            else:
                raise ValueError(
                    f"hidden_dims in tokenizer_kwargs must be a list or int, got {hidden_dims}"
                )
        else:
            if isinstance(hidden_dims, list):
                self.tokenizer_net = DenseResidualNet(
                    input_dim=self.num_features,
                    output_dim=output_dim,
                    hidden_dims=tuple(hidden_dims),
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                )
            elif isinstance(hidden_dims, int):
                self.tokenizer_net = MLP(
                    input_size=self.num_features,
                    hidden_size=hidden_dims,
                    output_size=output_dim,
                    activation_fn=activation,
                )
            else:
                raise ValueError(
                    f"hidden_dims in tokenizer_kwargs must be a list or int, got {hidden_dims}"
                )

    def forward(self, x: Tensor):
        """
        Parameters
        --------
        x: Tensor
            shape [batch_size, num_tokens, num_features]
                 =[batch_size, num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd].

        Returns
        --------
        x: Tensor
            shape [batch_size, num_tokens, d_model]
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Invalid shape for token embedding layer. "
                f"Expected last dimension to be {self.num_features}, got {tuple(x.shape[-1])}."
            )
        if self.individual_token_embedding:
            x = torch.stack(
                [
                    self.stack_tokenizer_nets[i](x[:, i, ...])
                    for i in range(self.num_tokens)
                ],
                dim=1,
            )
        else:
            x = self.tokenizer_net(x)

        return x


class PositionalEncoding(nn.Module):
    """
    A nn.Module that adds the positional encoding for f_min and f_max to each token.
    Two different types of positional encoding are implemented:
        * discrete positional encoding based on the definition in the paper "Attention is all you need"
        (https://arxiv.org/abs/1706.03762)
        * continuous positional encoding based on the definition in the paper "NeRF: Representing Scenes
        as Neural Radiance Fields for View Synthesis" (https://arxiv.org/abs/2003.08934) designed specifically
        for continuous position values.
    The implementation is adjusted to incorporate f_min and f_max.

    Methods
    -------
    forward:
        add the frequency encoding to the embedding matrix, dependent on f_min and f_max of each token
    """

    def __init__(self, d_model: int, positional_encoding_type: str):
        """
        Parameters:
        --------
        d_model: int
            size of transformer embedding dimension
        positional_encoding_type: str
            type of encoding, possibilities ["discrete", "continuous". "discrete+nn", "continuous+nn"]
            'discrete' for discrete positional encoding (from paper 'Attention is all you need'),
            'continuous' for continuous positional encoding (from paper
            'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis'

        """
        super(PositionalEncoding, self).__init__()

        self.num_min_vals = int(torch.ceil(torch.tensor(d_model) / 2))
        self.num_max_vals = d_model - self.num_min_vals

        self.positional_encoding_type = positional_encoding_type
        if "discrete" in positional_encoding_type:
            factor_min = torch.exp(
                -torch.arange(0, self.num_min_vals, 2)
                * math.log(10000.0)
                / self.num_min_vals
            )
            factor_max = torch.exp(
                -torch.arange(0, self.num_max_vals, 2)
                * math.log(10000.0)
                / self.num_max_vals
            )
        elif "continuous" in positional_encoding_type:
            factor_min = (
                torch.pow(2, torch.arange(0, self.num_min_vals, 2) / self.num_min_vals)
                * math.pi
            )
            factor_max = (
                torch.pow(2, torch.arange(0, self.num_max_vals, 2) / self.num_max_vals)
                * math.pi
            )
        else:
            raise ValueError(
                f"Invalid value for encoding_type.",
                f"Expected one of ['discrete', 'continuous'] to be in {self.encoding_type}.",
            )
        self.linear = None
        if "nn" in positional_encoding_type:
            self.linear = nn.Linear(d_model, d_model)
            self.initialize_linear()

        self.d_model = torch.tensor(d_model)

        self.register_buffer("factor_min", factor_min)
        self.register_buffer("factor_max", factor_max)

    def initialize_linear(self):
        self.linear.bias.data = torch.zeros_like(self.linear.bias.data)
        self.linear.weight.data = torch.eye(
            self.linear.weight.data.shape[0]
        ) + torch.normal(0, 0.01, size=self.linear.weight.data.shape)

    def forward(self, x: Tensor, position: Tensor) -> Tensor:
        """
        Parameters
        --------
        x: Tensor
            shape [batch_size, num_tokens, d_model]
        position: Tensor
            shape [batch_size, num_tokens, 3]
            last three dimensions correspond to [f_min, f_max, detector] of each token

        Returns
        --------
        x: Tensor
            same shape as input x
        """
        assert len(x.shape) == 3, (
            f"Invalid input shape in PositionalEncoding, expected len(input_dims) == 3, "
            f"got shape {x.shape})."
        )
        batch_size, num_tokens = x.shape[0], x.shape[1]
        assert self.d_model == x.shape[2], (
            f"Invalid input shape in PositionalEncoding, "
            f"expected x.shape[2] == {self.d_model}."
        )
        device = x.device
        min_vals, max_vals = position[..., 0], position[..., 1]

        # Assuming that min_val and max_val are the same for all blocks
        pos_embedding = torch.zeros(
            (batch_size, num_tokens, self.d_model), device=device
        )
        pos_embedding[..., 0 : self.num_min_vals : 2] = torch.sin(
            min_vals.unsqueeze(-1) * self.factor_min
        )
        pos_embedding[..., 1 : self.num_min_vals + 1 : 2] = torch.cos(
            min_vals.unsqueeze(-1) * self.factor_min
        )
        pos_embedding[..., self.num_min_vals : self.d_model : 2] = torch.sin(
            max_vals.unsqueeze(-1) * self.factor_max
        )
        f_max_cos_dim = pos_embedding[
            ..., self.num_min_vals + 1 : self.d_model : 2
        ].shape[-1]
        pos_embedding[..., self.num_min_vals + 1 : self.d_model : 2] = torch.cos(
            max_vals.unsqueeze(-1) * self.factor_max[:f_max_cos_dim]
        )
        if self.linear:
            pos_embedding = self.linear(pos_embedding)

        x += pos_embedding

        return x


class BlockEncoding(nn.Module):
    """
    A nn.Module that adds the embedding of different blocks to the input x.
    In the use case of GW, blocks refers to interferometers.

    Methods
    --------
    forward:
        obtains embedding of employed blocks and adds it to x.
    """

    def __init__(
        self, num_blocks: int, d_model: int, block_encoding_type: str = "sine"
    ):
        """
        Parameters
        --------
        num_blocks: int
            number of blocks
        d_model: int
            size of transformer embedding dimension
        """
        super(BlockEncoding, self).__init__()
        self.block_encoding = nn.Embedding(num_blocks, d_model)
        self.num_blocks = num_blocks
        self.d_model = d_model

        self.initialize_embedding()

    def initialize_embedding(self):
        pi = torch.tensor(math.pi)
        p_emb = torch.linspace(0, 2 * pi, self.d_model)
        off_set = torch.tensor(
            [i / self.num_blocks * 2 * pi for i in range(self.num_blocks)]
        )
        self.block_encoding.weight.data = torch.sin(p_emb + off_set.unsqueeze(1))

    def forward(self, x: Tensor, blocks: Tensor) -> Tensor:
        """
        Parameters
        --------
        x: Tensor
            shape [batch_size, num_tokens, d_model]
        blocks: Tensor
            shape [batch_size, num_tokens], specifying the block for each token.

        Returns
        --------
        x: Tensor
            shape [batch_size, num_tokens, d_model]
        """

        x = x + self.block_encoding(blocks.long())
        # x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) / 2

        return x


class MultiPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_vals: list, resolutions: list):
        super().__init__()
        num_encodings = len(max_vals)
        encoding_sizes = [((d_model // 2) // num_encodings) * 2] * num_encodings
        encoding_sizes[-1] += d_model - sum(encoding_sizes)
        assert sum(encoding_sizes) == d_model

        for i in range(num_encodings):
            k = torch.arange(0, encoding_sizes[i], 2)
            div_term = torch.exp(
                torch.log(torch.tensor(max_vals[i] / resolutions[i]))
                * (-k / encoding_sizes[i])
            )
            self.register_buffer("div_term_" + str(i), div_term)
        self.num_encodings = num_encodings

    def forward(self, x: Tensor, position: Tensor):
        """
        Parameters
        ----------
        x: Tensor, shape ``[batch_size, seq_length, embedding_dim]``
        position: Tensor, shape ``[batch_size, seq_length, self.num_encodings]``
        """
        position = position.unsqueeze(-1)
        start = 0
        pe = torch.zeros_like(x)
        for i in range(self.num_encodings):
            div_term = getattr(self, "div_term_" + str(i))
            end = start + 2 * len(div_term)
            pe[:, :, start:end:2] = torch.sin(position[:, :, i, :] * div_term)
            pe[:, :, start + 1 : end : 2] = torch.cos(position[:, :, i, :] * div_term)
            start = end
        return x + pe


class TransformerModel(nn.Module):
    """
    This nn.Module specifies the transformer encoder that can be used as an embedding network
    for the normalizing flow. Each raw token that refers to a specific frequency segment is embedded
    via TokenEmbedding. The frequency information of each raw token is added via FrequencyEncoding.
    Information about the blocks (=interferometers in GW use case) is included via BlockEmbedding.
    The output of the transformer encoder is reduced to the output dimension via a linear layer.

    Methods
    --------
    init_weights:
        initializes the weights uniformly and sets biases to zero
    forward:
        evaluates the transformer encoder model
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        positional_encoder: PositionalEncoding,
        block_encoder: BlockEncoding,
        final_net: DenseResidualNet,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        Parameters
        --------
        tokenizer: Tokenizer
        positional_encoder: PositionalEncoding
        block_encoder: BlockEncoding
        final_net: DenseResidualNet
        d_model: int
            embedding size of transformer
        dim_feedforward: int
            number of hidden dimensions in the feedforward neural networks of the transformer encoder
        nhead: int
            number of transformer heads
        num_layers: int
            number of transformer layers
        dropout: float
            dropout
        """
        super().__init__()
        self.model_type = "Transformer"

        self.tokenizer = tokenizer
        self.positional_encoding = positional_encoder
        self.block_encoding = block_encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

        self.final_net = final_net

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
        self,
        x: Tensor,
        position: Tensor,
        src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        """
        Parameters
        --------
        x: Tensor
            shape depends on which type of tokenization transform was used
            [batch_size, num_tokens, num_features] =
            [batch_size, num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
        position: Tensor
            shape [batch_size, num_blocks, 3], where the last dimension corresponds to [f_min, f_max, detector] per token
        src_key_padding_mask: Tensor
            shape [batch_size, num_tokens]

        Returns
        --------
        output: Tensor
            shape [batch_size, d_out_of_final_net if final_net else d_model]
        """
        if self.tokenizer is not None:
            x = self.tokenizer(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, position[..., 0:2])
        if self.block_encoding is not None:
            x = self.block_encoding(x, position[..., 2])

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Average over non-masked components.
        if src_key_padding_mask is not None:
            denominator = torch.sum(~src_key_padding_mask, -1, keepdim=True)
            x = torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2) / denominator
        else:
            x = torch.mean(x, dim=-2)

        if self.final_net is not None:
            x = self.final_net(x)

        return x


def create_transformer_enet(
    transformer_kwargs: dict,
    tokenizer_kwargs: dict = None,
    positional_encoder_kwargs: dict = None,
    block_encoder_kwargs: dict = None,
    final_net_kwargs: dict = None,
    added_context: bool = False,
):
    """
    Builder function for a transformer embedding network for complex 1D data
    with multiple blocks and channels.
    The complex signal has to be represented via the real part in channel 0 and
    the imaginary part in channel 1. Auxiliary signals may be contained in
    channels with indices => 2. In the GW use case, a block corresponds to a
    detector and channel 2 is used for ASD information.

    Parameters
    --------
    transformer_kwargs: dict
        settings for transformer
    tokenizer_kwargs: dict
        settings for tokenizer
    positional_encoder_kwargs: dict
        settings for positional encoder
    block_encoder_kwargs: dict
        settings for block encoder
    final_net_kwargs: dict
        settings for final network
    added_context: bool = False
        whether to add additional gnpe dimension to the context vector

    Returns
    --------
    model: TransformerModel

    """
    if added_context:
        raise ValueError(
            "GNPE is not yet implemented for transformer embedding network."
        )

    if tokenizer_kwargs is not None:
        tokenizer_kwargs["activation"] = torchutils.get_activation_function_from_string(
            tokenizer_kwargs["activation"]
        )
        tokenizer = Tokenizer(**tokenizer_kwargs)
    else:
        tokenizer = None

    if positional_encoder_kwargs is not None:
        positional_encoder_kwargs["d_model"] = transformer_kwargs["d_model"]
        positional_encoder = PositionalEncoding(**positional_encoder_kwargs)
    else:
        positional_encoder = None

    if block_encoder_kwargs is not None:
        block_encoder_kwargs["d_model"] = transformer_kwargs["d_model"]
        block_encoder = BlockEncoding(**block_encoder_kwargs)
    else:
        block_encoder = None

    if final_net_kwargs is not None:
        final_net_kwargs["activation"] = torchutils.get_activation_function_from_string(
            final_net_kwargs["activation"]
        )
        final_net = DenseResidualNet(**final_net_kwargs)
    else:
        final_net = None

    model = TransformerModel(
        tokenizer=tokenizer,
        positional_encoder=positional_encoder,
        block_encoder=block_encoder,
        final_net=final_net,
        **transformer_kwargs,
    )

    return model


class PoolingTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        positional_encoder,
        transformer_encoder,
        final_net=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.positional_encoder = positional_encoder
        self.transformer_encoder = transformer_encoder
        self.final_net = final_net

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
        self,
        src: torch.Tensor,
        position: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.tokenizer(src)

        if position is not None:
            # TODO: Update positional encoder to accept src_key_padding_mask.
            x = self.positional_encoder(x, position)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Average over non-masked components.
        if src_key_padding_mask is not None:
            denominator = torch.sum(~src_key_padding_mask, -1, keepdim=True)
            x = torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2) / denominator
        else:
            x = torch.mean(x, dim=-2)

        if self.final_net is not None:
            x = self.final_net(x)

        return x


def create_pooling_transformer(
    transformer_kwargs: dict,
    tokenizer_kwargs: dict = None,
    positional_encoder_kwargs: dict = None,
    final_net_kwargs: dict = None,
    added_context: bool = False,
):
    """Builder function for a transformer based multi-event encoder."""

    # build individual modules
    tokenizer_kwargs["activation"] = torchutils.get_activation_function_from_string(
        tokenizer_kwargs["activation"]
    )
    tokenizer = DenseResidualNet(**tokenizer_kwargs)
    positional_encoder = MultiPositionalEncoding(**positional_encoder_kwargs)
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=transformer_kwargs["d_model"],
        dim_feedforward=transformer_kwargs.get("dim_feedforward", 2048),
        nhead=transformer_kwargs["nhead"],
        dropout=transformer_kwargs.get("dropout", 0.1),
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(
        transformer_layer, num_layers=transformer_kwargs["num_layers"]
    )
    final_net_kwargs["activation"] = torchutils.get_activation_function_from_string(
        final_net_kwargs["activation"]
    )
    final_net = DenseResidualNet(**final_net_kwargs)

    encoder = PoolingTransformer(
        tokenizer,
        positional_encoder,
        transformer,
        final_net,
    )
    return encoder
