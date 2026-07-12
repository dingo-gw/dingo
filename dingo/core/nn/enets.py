"""Implementation of embedding networks.

Embedding networks registered with EMBEDDING_NETS follow a common contract:

* ``input_keys``: class attribute naming the batch entries the network consumes,
  in the order of its forward arguments.
* ``output_dim``: dimension of the embedded context vector.
* ``complete_settings(settings, sample_batch)``: classmethod inferring the
  network's own input dimensions from a sample batch; the completed settings
  (which must include ``output_dim``) are saved in the checkpoint, so loading
  never needs a data sample.
* ``init_data_spec()`` (optional): returns a dict describing the data variation
  the network wants for data-driven weight initialization (e.g. noise-free,
  un-formatted waveforms), or None if no initialization is needed.
* ``initialize_weights(batches, out_dir=None)`` (optional): consumes an iterator
  of batches matching the spec and initializes the network weights in-place.
  The trainer answers the spec with a matching dataloader and calls this hook;
  it does not know about specific architectures.

Context mergers registered with CONTEXT_MERGERS wrap an embedding network to mix
in the (standardized) context parameters; they follow the same contract, plus a
``merged_output_dim`` method used during settings completion.
"""

import os
from typing import Tuple, Union, List

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from dingo.core.SVD import SVDBasis
from dingo.core.nn.resnet import DenseResidualNet
from dingo.core.registry import CONTEXT_MERGERS, EMBEDDING_NETS
from dingo.core.utils import torchutils


class LinearProjectionRB(nn.Module):
    """
    A compression layer that reduces the input dimensionality via projection
    onto a reduced basis. The input data is of shape (batch_size, num_blocks,
    num_channels, num_bins). Each of the num_blocks blocks (for GW use case:
    block=detector) is treated independently.

    A single block consists of 1D data with num_bins bins (e.g. GW use case:
    num_bins=number of frequency bins). It has num_channels>=2 different
    channels, channel 0 and 1 store the real and imaginary part of the
    signal. Channels with index >=2 are used for auxiliary signals (such as
    PSD for GW use case).

    This layer compresses the complex signal in channels 0 and 1 to n_rb
    reduced-basis (rb) components. This is achieved by initializing the
    weights of this layer with the rb matrix V, such that the (2*n_rb)
    dimensional output of each block is the concatenation of the real and
    imaginary part of the reduced basis projection of the complex signal in
    channel 0 and 1. The projection of the auxiliary channels with index >=2
    onto these components is initialized with 0.

    Module specs
    --------
        input dimension:    (batch_size, num_blocks, num_channels, num_bins)
        output dimension:   (batch_size, 2 * n_rb * num_blocks)
    """

    def __init__(
        self,
        input_dims: List[int],
        n_rb: int,
        V_rb_list: Union[Tuple, None],
    ):
        """
        Parameters
        ----------
        input_dims : list
            dimensions of input batch, omitting batch dimension
            input_dims = [num_blocks, num_channels, num_bins]
        n_rb : int
            number of reduced basis elements used for projection
            the output dimension of the layer is 2 * n_rb * num_blocks
        V_rb_list : tuple of np.arrays, or None
            tuple with V matrices of the reduced basis SVD projection,
            convention for SVD matrix decomposition: U @ s @ V^h;
            if None, layer is not initialized with reduced basis projection,
            this is useful when loading a saved model
        """

        super(LinearProjectionRB, self).__init__()

        self.input_dims = input_dims
        self.num_blocks, self.num_channels, self.num_bins = self.input_dims
        self.n_rb = n_rb

        # define a linear projection layer for each block
        layers = []
        for _ in range(self.num_blocks):
            layers.append(nn.Linear(self.num_bins * self.num_channels, self.n_rb * 2))
        self.layers_rb = nn.ModuleList(layers)

        # initialize layers with reduced basis
        if V_rb_list is not None:
            if type(V_rb_list[0]) == str:
                V_rb_list = [np.load(el) for el in V_rb_list]
            self.test_dimensions(V_rb_list)
            self.init_layers(V_rb_list)

    @property
    def input_dim(self):
        return self.num_bins * self.num_channels * self.num_blocks

    @property
    def output_dim(self):
        return 2 * self.n_rb * self.num_blocks

    def test_dimensions(self, V_rb_list):
        """Test if input dimensions to this layer are consistent with each
        other, and the reduced basis matrices V."""
        if self.num_channels < 2:
            raise ValueError(
                "Number of channels needs to be at least 2, for real and "
                "imaginary parts."
            )
        if len(V_rb_list) != self.num_blocks:
            raise ValueError(
                "There must be exactly one reduced basis matrix V for each " "block."
            )
        for V in V_rb_list:
            if not isinstance(V, np.ndarray) or len(V.shape) != 2:
                raise ValueError(
                    "Reduced basis matrix V must be a numpy array with 2 axes."
                )
            if V.shape[0] != self.num_bins:
                raise ValueError(
                    "Number of input bins and number of rows in rb matrix V "
                    "need to match."
                )
            if V.shape[1] < self.n_rb:
                raise ValueError(
                    "More reduced basis elements requested than available."
                )

    def init_layers(self, V_rb_list):
        """
        Loop through layers and initialize them individually with the
        corresponding rb projection. V_rb_list is a list that contains the rb
        matrix V for each block. Each matrix V in V_rb_list is represented
        with a numpy array of shape (self.num_bins, num_el), where
        num_el >= self.n_rb.
        """
        n = self.n_rb
        k = self.num_bins
        for ind, layer in enumerate(self.layers_rb):
            V = V_rb_list[ind]

            # truncate V to n_rb basis elements
            V = V[:, :n]
            V_real, V_imag = (
                torch.from_numpy(V.real).float(),
                torch.from_numpy(V.imag).float(),
            )

            # initialize all weights and biases with zero
            layer.weight.data = torch.zeros_like(layer.weight.data)
            layer.bias.data = torch.zeros_like(layer.bias.data)

            # load matrix V into weights
            layer.weight.data[:n, :k] = torch.transpose(V_real, 1, 0)
            layer.weight.data[n:, :k] = torch.transpose(V_imag, 1, 0)
            layer.weight.data[:n, k : 2 * k] = -torch.transpose(V_imag, 1, 0)
            layer.weight.data[n:, k : 2 * k] = torch.transpose(V_real, 1, 0)

    def forward(self, x, **_):
        """RB projection. Additional kwargs (like context) are ignored."""
        if x.shape[1:] != (self.num_blocks, self.num_channels, self.num_bins):
            raise ValueError(
                f"Invalid shape for projection layer. "
                f"Expected {(self.num_blocks, self.num_channels, self.num_bins)}, "
                f"got {tuple(x.shape[1:])}."
            )
        out = []
        for ind in range(self.num_blocks):
            out.append(self.layers_rb[ind](x[:, ind, ...].flatten(start_dim=1)))
        x = torch.cat(out, dim=1)
        return x


@EMBEDDING_NETS.register("dense_svd")
class DenseSVDEmbedding(nn.Sequential):
    """
    The classic dingo embedding network: a linear projection onto a reduced (SVD)
    basis (LinearProjectionRB), followed by a dense residual network
    (DenseResidualNet). See the docstrings of the two modules for details.

    Consumes the "waveform" entry of the batch, of shape
    (batch_size, num_blocks, num_channels, num_bins). Subclasses nn.Sequential so
    that the state dict lays out as ("0.*", "1.*"), matching old checkpoints.
    """

    input_keys = ("waveform",)

    def __init__(
        self,
        input_dims: List[int],
        output_dim: int,
        hidden_dims: Tuple,
        svd: dict,
        activation: str = "elu",
        dropout: float = 0.0,
        batch_norm: bool = True,
        V_rb_list: Union[Tuple, None] = None,
    ):
        """
        Parameters
        ----------
        input_dims : list
            dimensions of input batch, omitting batch dimension,
            input_dims = [num_blocks, num_channels, num_bins]. Inferred from a
            sample batch by complete_settings; not a user setting.
        output_dim : int
            output dimension (dimension of the embedded context)
        hidden_dims : tuple
            dimensions of the hidden layers of the residual network
        svd : dict
            SVD settings; "size" is the number of reduced-basis elements used for
            the projection (further entries are consumed by the training pipeline
            when generating the SVD).
        activation : str
            activation function used in the residual blocks
        dropout : float
            dropout probability in the residual blocks, for regularization
        batch_norm : bool
            whether to use batch normalization
        V_rb_list : tuple of np.arrays, or None
            V matrices of the SVD projection used to initialize the projection
            weights directly. Usually None: the projection is seeded via the
            initialize_weights hook instead.
        """
        projection = LinearProjectionRB(input_dims, svd["size"], V_rb_list)
        resnet = DenseResidualNet(
            input_dim=projection.output_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=torchutils.get_activation_function_from_string(activation),
            dropout=dropout,
            batch_norm=batch_norm,
        )
        super().__init__(projection, resnet)
        self.output_dim = output_dim
        self.svd_settings = svd

    @classmethod
    def complete_settings(cls, settings: dict, sample_batch: dict) -> dict:
        """Infer input_dims from the sample batch; return completed settings."""
        if "input_dims" in settings:
            raise ValueError(
                "'input_dims' is derived from the data and must not be specified "
                "in the embedding net settings."
            )
        return {**settings, "input_dims": list(sample_batch["waveform"].shape)}

    def init_data_spec(self):
        """
        Data variation for seeding the SVD projection: clean (noise-free),
        un-formatted waveforms at a fixed reference luminosity distance. Returns
        None if the svd settings do not request seeding (no
        num_training_samples), e.g. when loading a saved model.
        """
        if "num_training_samples" not in self.svd_settings:
            return None
        num_samples = self.svd_settings["num_training_samples"] + self.svd_settings.get(
            "num_validation_samples", 0
        )
        return {
            "noise": False,
            "network_format": False,
            "fix_parameters": {"luminosity_distance": 100.0},
            "num_samples": num_samples,
        }

    def initialize_weights(self, batches, out_dir=None):
        """
        Seed the projection layer with an SVD basis built from clean waveforms.

        Parameters
        ----------
        batches : iterable
            Batches matching init_data_spec: dicts with "waveform" a
            {block: (batch_size, len) complex array} dict (in the GW use case, a
            block is a detector) and "parameters", used for validation
            diagnostics. Iteration stops once num_samples have been consumed.
        out_dir : str = None
            If provided, SVD validation diagnostics are computed and saved here.
        """
        svd = self.svd_settings
        num_training = svd["num_training_samples"]
        num_validation = svd.get("num_validation_samples", 0)
        num_samples = num_training + num_validation

        waveforms = None
        parameters = pd.DataFrame()
        collected = 0
        for batch in batches:
            strain_data = batch["waveform"]
            if waveforms is None:
                waveforms = {
                    block: np.empty(
                        (num_samples, strains.shape[-1]), dtype=np.complex128
                    )
                    for block, strains in strain_data.items()
                }
            batch_size = len(next(iter(strain_data.values())))
            n = min(batch_size, num_samples - collected)
            parameters = pd.concat(
                [parameters, pd.DataFrame(batch["parameters"]).iloc[:n]],
                ignore_index=True,
            )
            for block, strains in strain_data.items():
                waveforms[block][collected : collected + n] = strains[:n]
            collected += n
            if collected == num_samples:
                break
        if collected < num_samples:
            raise IndexError(
                f"Requested {num_samples} samples for SVD initialization, but the "
                f"dataloader only provided {collected}."
            )

        projection = self[0]
        V_rb_list = []
        for block, data in waveforms.items():
            print(f"Generating SVD basis for block {block}.")
            basis = SVDBasis()
            basis.generate_basis(data[:num_training], svd["size"])
            if out_dir is not None and num_validation > 0:
                basis.compute_test_mismatches(
                    data[num_training:],
                    parameters=parameters.iloc[num_training:].reset_index(drop=True),
                    verbose=True,
                )
                basis.to_file(os.path.join(out_dir, f"svd_{block}.hdf5"))
            # The provided waveforms may be longer than the network input (leading
            # entries outside the network's frequency range). These must be zero,
            # and the corresponding rows of V are dropped.
            V = basis.V
            excess = len(V) - projection.num_bins
            if not np.allclose(V[:excess], 0):
                raise ValueError(
                    f"Block {block}: SVD basis has non-zero entries outside the "
                    f"network input range (waveform length {len(V)}, network "
                    f"num_bins {projection.num_bins})."
                )
            V_rb_list.append(V[excess:])

        projection.test_dimensions(V_rb_list)
        projection.init_layers(V_rb_list)


class ModuleMerger(nn.Module):
    """
    This is a wrapper used to process multiple different kinds of context
    information collected in x = (x_0, x_1, ...). For each kind of context
    information x_i, an individual embedding network is provided in
    enets = (enet_0, enet_1, ...). The embedded output of the forward method
    is the concatenation of the individual embeddings enet_i(x_i).

    In the GW use case, this wrapper can be used to embed the
    high-dimensional signal input into a lower dimensional feature vector
    with a large embedding network, while applying an identity embedding to
    the time shifts.

    Module specs
    --------
        input dimension:    (batch_size, ...), (batch_size, ...), ...
        output dimension:   (batch_size, ?)
    """

    def __init__(
        self,
        module_list: Tuple,
    ):
        """
        Parameters
        ----------
        module_list : tuple
            nn.Modules for embedding networks,
            use torch.nn.Identity for identity mappings
        """
        super(ModuleMerger, self).__init__()
        self.enets = nn.ModuleList(module_list)

    def forward(self, *x):
        if len(x) != len(self.enets):
            raise ValueError("Invalid number of input tensors provided.")
        x = [module(xi) for module, xi in zip(self.enets, x)]
        return torch.cat(x, axis=1)


@CONTEXT_MERGERS.register("concat")
class ConcatContextMerger(ModuleMerger):
    """
    Default context merger: concatenates the embedded data with the (standardized)
    context parameters, e.g. GNPE proxies. Wraps the embedding network and an
    identity map via ModuleMerger, which keeps the state-dict layout of old
    checkpoints ("enets.0.*").
    """

    def __init__(self, embedding_net: nn.Module, num_context_parameters: int):
        """
        Parameters
        ----------
        embedding_net : nn.Module
            The wrapped embedding network.
        num_context_parameters : int
            Number of context parameters concatenated to the embedded data.
            Inferred from a sample batch during settings completion.
        """
        super().__init__((embedding_net, nn.Identity()))
        self.input_keys = (*embedding_net.input_keys, "context_parameters")
        self.output_dim = embedding_net.output_dim + num_context_parameters

    def forward(self, *x):
        # Unlike ModuleMerger, the wrapped embedding may consume several inputs
        # (e.g. the transformer: waveform, position, mask); the context parameters
        # are always the last one.
        *data, context = x
        return torch.cat([self.enets[0](*data), self.enets[1](context)], dim=1)

    @staticmethod
    def merged_output_dim(embedding_output_dim: int, num_context_parameters: int):
        """Output dimension of the merged embedding, for settings completion."""
        return embedding_output_dim + num_context_parameters


@CONTEXT_MERGERS.register("mlp")
class MLPContextMerger(nn.Module):
    """
    Context merger that mixes the embedded data and the (standardized) context
    parameters through a learned MLP, in contrast to the concat merger which
    simply concatenates them. Ported from the chained-NPE branch
    (ContextMergerMLP).

    The data is first embedded, z = embedding_net(x). The context parameters c
    are concatenated with z and passed through a DenseResidualNet M, producing
    z_new = M(concat(z, c)) with dim(z_new) = output_dim. By default output_dim
    equals dim(z), so the conditioning context fed to the downstream flow does
    not grow with the number of context parameters.
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        num_context_parameters: int,
        hidden_dims: Tuple,
        output_dim: int = None,
        activation: str = "elu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Parameters
        ----------
        embedding_net : nn.Module
            The wrapped embedding network.
        num_context_parameters : int
            Number of context parameters mixed into the embedded data. Inferred
            from a sample batch during settings completion.
        hidden_dims : tuple
            dimensions of the hidden layers of the merging DenseResidualNet
        output_dim : int = None
            output dimension of the merged embedding; defaults to the output
            dimension of the wrapped embedding network
        activation : str
            activation function used in the residual blocks
        dropout : float
            dropout probability in the residual blocks
        batch_norm : bool
            whether to use batch normalization
        """
        super().__init__()
        if output_dim is None:
            output_dim = embedding_net.output_dim
        self.embedding_net = embedding_net
        self.context_module = DenseResidualNet(
            input_dim=embedding_net.output_dim + num_context_parameters,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=torchutils.get_activation_function_from_string(activation),
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.input_keys = (*embedding_net.input_keys, "context_parameters")
        self.output_dim = output_dim

    def forward(self, *x):
        *data, context = x
        z = self.embedding_net(*data)
        return self.context_module(torch.cat([z, context], dim=1))

    @staticmethod
    def merged_output_dim(embedding_output_dim: int, output_dim: int = None, **_unused):
        """Output dimension of the merged embedding, for settings completion."""
        return output_dim if output_dim is not None else embedding_output_dim


def create_enet_with_projection_layer_and_dense_resnet(
    input_dims: List[int],
    # n_rb: int,
    V_rb_list: Union[Tuple, None],
    output_dim: int,
    hidden_dims: Tuple,
    svd: dict,
    activation: str = "elu",
    dropout: float = 0.0,
    batch_norm: bool = True,
    added_context: bool = False,
):
    """
    Builder function for 2-stage embedding network for 1D data with multiple
    blocks and channels. Module 1 is a linear layer initialized as the
    projection of the complex signal onto reduced basis components via the
    LinearProjectionRB, where the blocks are kept separate. See docstring
    of LinearProjectionRB for details. Module 2 is a sequence of dense residual
    layers, that is used to further reduce the dimensionality.

    The projection requires the complex signal to be represented via the real
    part in channel 0 and the imaginary part in channel 1. Auxiliary signals
    may be contained in channels with indices => 2. In GW use case a block
    corresponds to a detector and channel 2 is used for ASD information.

    If added_context = True, the 2-stage embedding network described above is
    merged with an identity mapping via ModuleMerger. Then, the expected input
    is not x with x.shape = (batch_size, num_blocks, num_channels, num_bins),
    but rather the tuple *(x, z), where z is additional context information. The
    output of the full module is then the concatenation of enet(x) and z. In
    GW use case, this is used to concatenate the applied time shifts z to the
    embedded feature vector of the strain data enet(x).

    Module specs
    --------
    For added_context == False:
        input dimension:    (batch_size, num_blocks, num_channels, num_bins)
        output dimension:   (batch_size, output_dim)
    For added_context == True:
        input dimension:    (batch_size, num_blocks, num_channels, num_bins),
                            (batch_size, N)
        output dimension:   (batch_size, output_dim + N)

    :param input_dims:  list
        dimensions of input batch, omitting batch dimension
        input_dims = (num_blocks, num_channels, num_bins)
    :param n_rb: int
        number of reduced basis elements used for projection
        the output dimension of the layer is 2 * n_rb * num_blocks
    :param V_rb_list: tuple of np.arrays, or None
        tuple with V matrices of the reduced basis SVD projection,
        convention for SVD matrix decomposition: U @ s @ V^h;
        if None, layer is not initialized with reduced basis projection,
        this is useful when loading a saved model
    :param output_dim: int
        output dimension of the full module
    :param hidden_dims: tuple
        tuple with dimensions of hidden layers of module 2
    :param activation: str
        str that specifies activation function used in residual blocks
    :param dropout: float
        dropout probability for residual blocks used for reqularization
    :param batch_norm: bool
        flag that specifies whether to use batch normalization
    :param added_context: bool
        if set to True, additional context z is concatenated to the embedded
        feature vector enet(x); note that in this case, the expected input is
        a tuple with 2 elements, input = (x, z) rather than just the tensor x.
    :return: nn.Module
    """
    enet = DenseSVDEmbedding(
        input_dims=input_dims,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        svd=svd,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        V_rb_list=V_rb_list,
    )

    if not added_context:
        return enet
    else:
        return ModuleMerger((enet, nn.Identity()))


if __name__ == "__main__":
    pass
