from typing import List
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        emb_size: int,
        individual_token_embedding: bool = False,
    ):
        """
        Parameters
        --------
        input_dims: List[int]
            containing [num_blocks, num_channels, num_tokens, num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
        emb_size: int
            size of embedding dimension
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
            self.stack_linear = [
                nn.Linear(
                    self.num_channels * self.num_bins_per_token, emb_size, bias=False
                )
                for _ in range(self.num_tokens)
            ]
        else:
            self.linear = nn.Linear(
                self.num_channels * self.num_bins_per_token, emb_size, bias=False
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
                if self.individual_token_embedding:
                    lin_layer = self.stack_linear[i]
                else:
                    lin_layer = self.linear
                # extract tensor of shape [batch_size, num_channels, num_bins_per_token]
                x_b = x[:, b, :, i, :]
                out_b.append(lin_layer(x_b.flatten(start_dim=1)))
            out.append(
                torch.stack(out_b, dim=2).reshape([-1, self.num_tokens, self.emb_size])
            )
        x = torch.stack(out, dim=3).reshape(
            [-1, self.num_blocks, self.num_tokens, self.emb_size]
        )

        return x * math.sqrt(self.emb_size)


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

    def __init__(self, emb_size: int, encoding_type: str, dropout: float):
        """
        Parameters:
        --------
        emb_size: int
            size of embedding dimension
        encoding_type: str
            type of encoding, possibilities ["discrete", "continuous"]
        dropout: float
            dropout value
        """
        super(FrequencyEncoding, self).__init__()

        self.emb_size = torch.tensor(emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type

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

        Returns
        --------
        x: Tensor
            same shape as input x
        """
        batch_size, num_tokens = x.shape[0], x.shape[2]
        assert self.emb_size == x.shape[3]
        emb_size_f_min = int(self.emb_size / 2)
        emb_size_f_max = self.emb_size - emb_size_f_min

        if self.encoding_type == "discrete":
            d_f_min = torch.exp(
                -torch.arange(0, emb_size_f_min, 2) * math.log(10000.0) / emb_size_f_min
            )
            d_f_max = torch.exp(
                -torch.arange(0, emb_size_f_max, 2) * math.log(10000.0) / emb_size_f_max
            )
        elif self.encoding_type == "continuous":
            d_f_min = torch.pow(2, torch.arange(0, emb_size_f_min, 2)) * math.pi
            d_f_max = torch.pow(2, torch.arange(0, emb_size_f_max, 2)) * math.pi
        else:
            raise ValueError(
                f"Invalid value for encoding_type.",
                f"Expected one of ['discrete', 'continuous'], got {self.encoding_type}.",
            )

        if d_f_min.shape[0] + d_f_max.shape[0] != self.emb_size:
            d_f_max = d_f_max[: int(emb_size_f_max / 2)]

        # Assuming that f_min and f_max are the same for all blocks
        pos_embedding = torch.zeros((batch_size, 1, num_tokens, self.emb_size))
        pos_embedding[:, :, :, 0:emb_size_f_min:2] = torch.sin(
            f_min.reshape(batch_size, 1, num_tokens, 1) * d_f_min
        )
        pos_embedding[:, :, :, 1 : emb_size_f_min + 1 : 2] = torch.cos(
            f_min.reshape(batch_size, 1, num_tokens, 1) * d_f_min
        )
        pos_embedding[:, :, :, emb_size_f_min + 1 : self.emb_size : 2] = torch.sin(
            f_max.reshape(batch_size, 1, num_tokens, 1) * d_f_max
        )
        pos_embedding[:, :, :, emb_size_f_min + 2 : self.emb_size : 2] = torch.cos(
            f_max.reshape(batch_size, 1, num_tokens, 1) * d_f_max
        )

        x = x + pos_embedding

        return self.dropout(x)


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
        self.encoding = nn.Embedding(num_blocks, emb_size)
        self.num_blocks = num_blocks
        self.emb_size = emb_size

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
        block_encoding = self.encoding(blocks.long()) * math.sqrt(self.emb_size)
        out = []
        for i in range(self.num_blocks):
            block_out = x[:, i, :, :] + block_encoding[:, i, :].reshape(
                block_encoding.shape[0], 1, -1
            )
            out.append(block_out)
        x = torch.cat(out, dim=1)

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
        emb_size: int,
        num_head: int,
        num_layers: int,
        d_hid: int,
        individual_token_embedding: bool,
        frequency_encoding_type: str,
        dropout: float = 0.5,
    ):
        """
        Parameters
        --------
        input_dims: List[int]
            containing [num_blocks, num_channels, num_tokens, num_bins_per_token]
            where num_blocks = number of interferometers in GW use case, and num_channels = [real, imag, asd]
        d_out: int
            dimension of output
        emb_size: int
            size of embedding dimension
        num_head: int
            number of transformer heads
        num_layers: int
            number of transformer layers
        d_hid: int
            number of hidden dimensions in the feedforward neural networks of the transformer encoder
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
            emb_size=emb_size,
            individual_token_embedding=self.individual_token_embedding,
        )
        self.freq_encoder = FrequencyEncoding(
            emb_size=emb_size, encoding_type=frequency_encoding_type, dropout=dropout
        )
        self.block_encoder = BlockEmbedding(
            num_blocks=self.num_blocks, emb_size=emb_size
        )
        encoder_layers = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_head,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(emb_size * self.num_tokens * self.num_blocks, d_out)

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize weights of transformer encoder to uniform values and set biases to zero.
        """
        init_range = 0.1
        if self.individual_token_embedding:
            for i in range(self.num_tokens):
                self.embedding.stack_linear[i].weight.data.uniform_(
                    -init_range, init_range
                )
        else:
            self.embedding.linear.weight.data.uniform(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

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
        src = self.block_encoder(src, blocks)

        if src_mask is None:
            output = self.transformer_encoder(src)
        else:
            output = self.transformer_encoder(src, src_mask)
        output = self.flatten(output)
        output = self.linear(output)

        return output
