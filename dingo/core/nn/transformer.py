from types import SimpleNamespace
from typing import Callable, List, Tuple, Union
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dingo.core.nn.resnet import DenseResidualNet


class MLP(nn.Module):
    """Simple MLP with one hidden layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation_fn: Callable,
    ):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.activation = activation_fn
        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear0(x))
        x = self.activation(self.linear1(x))
        return x


class TokenEmbedding(nn.Module):
    """
    A nn.Module that maps each frequency fragment of length num_bins_per_token and width num_channels
    into an embedding vector of fixed emb_size via (the same or individual) linear layers.

    Methods
    -------
    forward:
        obtain the token embedding for a Tensor of shape
        [batch_size, num_blocks, num_channels, num_tokens, num_bins_per_token]
        where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dims: Union[List, int],
        emb_size: int,
        activation: Callable,
        dropout: float,
        batch_norm: bool,
        layer_norm: bool,
        individual_token_embedding: bool = False,
    ):
        """
        Parameters
        --------
        input_dims: List[int]
            containing [num_blocks, num_channels, num_tokens, num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
        hidden_dims: Optional[Tuple, int]
            if type List: dimensions of hidden layers for DenseResidualNet
            if type int: number of nodes in linear layer
        emb_size: int
            size of embedding dimension
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
        super(TokenEmbedding, self).__init__()

        (
            self.num_blocks,
            self.num_channels,
            self.num_tokens,
            self.num_bins_per_token,
        ) = input_dims
        if individual_token_embedding:
            if type(hidden_dims) is list:
                self.stack_embedding_networks = nn.ModuleList(
                    [
                        DenseResidualNet(
                            input_dim=self.num_channels * self.num_bins_per_token,
                            output_dim=emb_size,
                            hidden_dims=tuple(hidden_dims),
                            activation=activation,
                            dropout=dropout,
                            batch_norm=batch_norm,
                            layer_norm=layer_norm,
                        )
                        for _ in range(self.num_tokens)
                    ]
                )
            else:
                assert type(hidden_dims) is int
                self.stack_embedding_networks = nn.ModuleList(
                    [
                        MLP(
                            self.num_channels * self.num_bins_per_token,
                            hidden_dims,
                            emb_size,
                            activation,
                        )
                        for _ in range(self.num_tokens)
                    ]
                )
        else:
            if type(hidden_dims) is list:
                self.embedding_networks = DenseResidualNet(
                    input_dim=self.num_channels * self.num_bins_per_token,
                    output_dim=emb_size,
                    hidden_dims=tuple(hidden_dims),
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                )
            else:
                assert type(hidden_dims) is int
                self.embedding_networks = MLP(
                    self.num_channels * self.num_bins_per_token,
                    hidden_dims,
                    emb_size,
                    activation,
                )

        self.individual_token_embedding = individual_token_embedding
        self.emb_size = emb_size

    def forward(self, x: Tensor):
        """
        Parameters
        --------
        x: Tensor
            shape [batch_size, num_blocks, num_channels, num_tokens, num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]

        Returns
        --------
        x: Tensor
            shape [batch_size, num_blocks, num_tokens, emb_size]
        """
        if x.shape[1:] != (
            self.num_blocks,
            self.num_channels,
            self.num_tokens,
            self.num_bins_per_token,
        ):
            raise ValueError(
                f"Invalid shape for token embedding layer. "
                f"Expected {(self.num_blocks, self.num_channels, self.num_tokens, self.num_bins_per_token)}, "
                f"got {tuple(x.shape[1:])}."
            )
        out = []
        # TODO: Rewrite with torch.vmap for more efficiency
        for b in range(self.num_blocks):
            out_b = []
            for i in range(self.num_tokens):
                # apply linear layer for tensor of shape [batch_size, num_channels, num_bins_per_token]
                if self.individual_token_embedding:
                    out_b.append(
                        self.stack_embedding_networks[i](
                            x[:, b, :, i, :].flatten(start_dim=1)
                        )
                    )
                else:
                    out_b.append(
                        self.embedding_networks(x[:, b, :, i, :].flatten(start_dim=1))
                    )
            out.append(torch.stack(out_b, dim=1))
        x = torch.stack(out, dim=1)

        return x


class FrequencyEncoding(nn.Module):
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

    def __init__(self, emb_size: int, encoding_type: str):
        """
        Parameters:
        --------
        emb_size: int
            size of embedding dimension
        encoding_type: str
            type of encoding, possibilities ["discrete", "continuous". "discrete+nn", "continuous+nn"]
        """
        super(FrequencyEncoding, self).__init__()

        self.emb_size_f_min = int(torch.ceil(torch.tensor(emb_size) / 2))
        self.emb_size_f_max = emb_size - self.emb_size_f_min

        self.encoding_type = encoding_type
        if "discrete" in encoding_type:
            d_f_min = torch.exp(
                -torch.arange(0, self.emb_size_f_min, 2)
                * math.log(10000.0)
                / self.emb_size_f_min
            )
            d_f_max = torch.exp(
                -torch.arange(0, self.emb_size_f_max, 2)
                * math.log(10000.0)
                / self.emb_size_f_max
            )
        elif "continuous" in encoding_type:
            d_f_min = (
                torch.pow(
                    2, torch.arange(0, self.emb_size_f_min, 2) / self.emb_size_f_min
                )
                * math.pi
            )
            d_f_max = (
                torch.pow(
                    2, torch.arange(0, self.emb_size_f_max, 2) / self.emb_size_f_max
                )
                * math.pi
            )
        else:
            raise ValueError(
                f"Invalid value for encoding_type.",
                f"Expected one of ['discrete', 'continuous'] to be in {self.encoding_type}.",
            )
        self.linear = None
        if "nn" in encoding_type:
            self.linear = nn.Linear(emb_size, emb_size)
            self.initialize_linear()

        self.emb_size = torch.tensor(emb_size)

        self.register_buffer("d_f_min", d_f_min)
        self.register_buffer("d_f_max", d_f_max)

    def initialize_linear(self):
        self.linear.bias.data = torch.zeros_like(self.linear.bias.data)
        self.linear.weight.data = torch.eye(
            self.linear.weight.data.shape[0]
        ) + torch.normal(0, 0.01, size=self.linear.weight.data.shape)

    def forward(self, x: Tensor, f_min: Tensor, f_max: Tensor) -> Tensor:
        """
        Parameters
        --------
        x: Tensor
            shape [batch_size, num_blocks, num_tokens, emb_size]
        f_min: Tensor
            shape [batch_size, num_tokens]
        f_max: Tensor
            shape [batch_size, num_tokens]
        device: str
            current device

        Returns
        --------
        x: Tensor
            same shape as input x
        """
        batch_size, num_tokens = x.shape[0], x.shape[2]
        assert self.emb_size == x.shape[3]
        device = x.device

        # Assuming that f_min and f_max are the same for all blocks
        pos_embedding = torch.zeros(
            (batch_size, 1, num_tokens, self.emb_size), device=device
        )
        pos_embedding[:, :, :, 0 : self.emb_size_f_min : 2] = torch.sin(
            f_min.reshape(batch_size, 1, num_tokens, 1) * self.d_f_min
        )
        pos_embedding[:, :, :, 1 : self.emb_size_f_min + 1 : 2] = torch.cos(
            f_min.reshape(batch_size, 1, num_tokens, 1) * self.d_f_min
        )
        pos_embedding[:, :, :, self.emb_size_f_min : self.emb_size : 2] = torch.sin(
            f_max.reshape(batch_size, 1, num_tokens, 1) * self.d_f_max
        )
        f_max_cos_dim = pos_embedding[
            :, :, :, self.emb_size_f_min + 1 : self.emb_size : 2
        ].shape[3]
        pos_embedding[:, :, :, self.emb_size_f_min + 1 : self.emb_size : 2] = torch.cos(
            f_max.reshape(batch_size, 1, num_tokens, 1) * self.d_f_max[:f_max_cos_dim]
        )
        if self.linear:
            tmp = []
            for i in range(num_tokens):
                tmp.append(self.linear(pos_embedding[:, :, i, :]))
            pos_embedding = torch.stack(tmp, dim=2)

        x = (x + pos_embedding) / 2

        return x


class BlockEmbedding(nn.Module):
    """
    A nn.Module that adds the embedding of different blocks to the input x.
    In the use case of GW, blocks refers to interferometers.

    Methods
    --------
    forward:
        obtains embedding of employed blocks and adds it to x.
    """

    def __init__(self, num_blocks: int, emb_size: int):
        """
        Parameters
        --------
        num_blocks: int
            number of blocks
        emb_size: int
            size of embedding dimension
        """
        super(BlockEmbedding, self).__init__()
        self.block_embedding = nn.Embedding(num_blocks, emb_size)
        self.num_blocks = num_blocks
        self.emb_size = emb_size

        self.initialize_embedding()

    def initialize_embedding(self):
        pi = torch.tensor(math.pi)
        p_emb = torch.linspace(0, 2 * pi, self.emb_size)
        off_set = torch.tensor(
            [i / self.num_blocks * 2 * pi for i in range(self.num_blocks)]
        )
        self.block_embedding.weight.data = torch.sin(p_emb + off_set.unsqueeze(1))

    def forward(self, x: Tensor, blocks: Tensor) -> Tensor:
        """
        Parameters
        --------
        x: Tensor
            shape [batch_size, num_blocks, num_tokens, emb_size]
        blocks: Tensor
            shape [batch_size, num_blocks], specifying the order of blocks in dim num_blocks of x.

        Returns
        --------
        x: Tensor
            shape [batch_size, num_blocks*num_tokens, emb_size]
        """
        if blocks.shape[1] != self.num_blocks:
            raise ValueError(
                f"Invalid shape for block indices in block encoding layer. "
                f"Expected {self.num_blocks}, got {len(blocks)}."
            )
        if x.shape[1] != self.num_blocks:
            raise ValueError(
                f"Invalid input shape in block encoding layer. "
                f"Expected {self.num_blocks}, got {x.shape[1]}."
            )

        x = x + torch.unsqueeze(self.block_embedding(blocks.long()), 2)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) / 2

        return x


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
        input_dims: List[int],
        d_out: int,
        num_head: int,
        num_layers: int,
        d_hid: int,
        hidden_dims_token_embedding: Union[int, Tuple],
        activation: Callable,
        batch_norm: bool,
        layer_norm: bool,
        individual_token_embedding: bool,
        frequency_encoding_type: str,
        dropout: float = 0.1,
    ):
        """
        Parameters
        --------
        input_dims: List[int]
            containing [num_blocks, num_channels, num_tokens, num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
        d_out: int
            dimension of output, corresponds to embedding size of transformer
        num_head: int
            number of transformer heads
        num_layers: int
            number of transformer layers
        d_hid: int
            number of hidden dimensions in the feedforward neural networks of the transformer encoder
        hidden_dims_token_embedding: Union[int, Tuple]
            if int: dimension of linear layer
            if Tuple: dimensions of hidden layers of DenseResNet used in TokenEmbedding
        activation: Callable
            activation function used in TokenEmbedding
        batch_norm: bool
            whether to apply batch normalization in the tokenizer
        layer_norm: bool
            whether to apply layer normalization in the tokenizer
        individual_token_embedding: bool
            whether to embed each raw token with an individual embedding network or not
        frequency_encoding_type: str
            type of frequency encoding, either 'discrete' for discrete positional encoding (from paper
            'Attention is all you need') or 'continuous' for continuous positional encoding (from paper
            'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis'
        dropout: float
            dropout
        """
        super().__init__()
        self.model_type = "Transformer"
        self.individual_token_embedding = individual_token_embedding

        (
            self.num_blocks,
            self.num_channels,
            self.num_tokens,
            self.num_bins_per_token,
        ) = input_dims
        self.embedding = TokenEmbedding(
            input_dims=input_dims,
            emb_size=d_out,
            hidden_dims=hidden_dims_token_embedding,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            individual_token_embedding=self.individual_token_embedding,
        )
        self.freq_encoder = FrequencyEncoding(
            emb_size=d_out, encoding_type=frequency_encoding_type
        )
        self.block_embedding = BlockEmbedding(
            num_blocks=self.num_blocks, emb_size=d_out
        )
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_out,
            nhead=num_head,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.adapt_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.d_out = d_out

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
        src: Tensor,
        blocks: Tensor,
        f_min: Tensor,
        f_max: Tensor,
        src_mask: Tensor = None,
    ) -> Tensor:
        """
        Parameters
        --------
        src: Tensor
            shape [batch_size, num_blocks, num_channels, num_tokens, num_bins_per_token]
        blocks: Tensor
            shape [batch_size, num_blocks], determines the ordering of the blocks in BlockEmbedding
        f_min: Tensor
            shape [batch_size, num_tokens]
        f_max: Tensor
            shape [batch_size, num_tokens]
        src_mask: Tensor
            shape [batch_size, num_tokens, num_tokens]

        Returns
        --------
        output: Tensor
            shape [batch_size, d_out]
        """
        src = self.embedding(src)
        src = self.freq_encoder(src, f_min, f_max)
        src = self.block_embedding(src, blocks)
        src = self.dropout(src)

        if src_mask is None:
            output = self.transformer_encoder(src)
        else:
            output = self.transformer_encoder(src, src_mask)
        output = self.adapt_avg_pool(
            output.reshape(output.shape[0], self.d_out, -1)
        ).squeeze()

        return output


class MultiPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_vals, resolutions):
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


class PoolingTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        positional_encoder,
        transformer_encoder,
        final_net=None,
        extra_skip=False,
        extra_skip_2=False,
        extra_skip_3=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.positional_encoder = positional_encoder
        self.transformer_encoder = transformer_encoder
        self.final_net = final_net
        self.extra_skip = extra_skip
        self.extra_skip_2 = extra_skip_2
        self.extra_skip_3 = extra_skip_3

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

        if self.extra_skip or self.extra_skip_2:
            # Skip is mean (across tokens) of x right after the tokenizer.
            if src_key_padding_mask is not None:
                denominator = torch.sum(~src_key_padding_mask, -1, keepdim=True)
                skip = (
                    torch.sum(x * ~src_key_padding_mask.unsqueeze(-1), dim=-2)
                    / denominator
                )
            else:
                skip = torch.mean(x, dim=-2)

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

        if self.extra_skip:
            x = x + skip
        if self.extra_skip_2:
            x = torch.cat((x, skip), dim=-1)

        if self.final_net is not None:
            x = self.final_net(x)

        if self.extra_skip_3:
            # Skip is mean (across tokens) of initial input.
            if src_key_padding_mask is not None:
                denominator = torch.sum(~src_key_padding_mask, -1, keepdim=True)
                skip = (
                    torch.sum(src * ~src_key_padding_mask.unsqueeze(-1), dim=-2)
                    / denominator
                )
            else:
                skip = torch.mean(src, dim=-2)
            x = torch.cat((x, skip), dim=-1)

        return x


def create_pooling_transformer(config):
    """Builder function for a transformer based multi-event encoder."""
    # autocomplete config
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    # config.tokenizer["input_dim"] = config.d_dingo_encoding
    config.positional_encoder["d_model"] = config.transformer["d_model"]
    config.tokenizer["output_dim"] = config.transformer["d_model"]
    config.final_net["input_dim"] = config.transformer["d_model"]

    # Experiment with extra skip connections for boosting performance and avoiding
    # local minima.
    extra_skip = config.transformer.get("extra_skip", False)
    extra_skip_2 = config.transformer.get("extra_skip_2", False)
    extra_skip_3 = config.transformer.get("extra_skip_3", False)
    if extra_skip_2:
        config.final_net["input_dim"] *= 2

    # build individual modules
    tokenizer = DenseResidualNet(**config.tokenizer)
    positional_encoder = MultiPositionalEncoding(**config.positional_encoder)
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

    encoder = PoolingTransformer(
        tokenizer,
        positional_encoder,
        transformer,
        final_net,
        extra_skip,
        extra_skip_2,
        extra_skip_3,
    )
    return encoder
