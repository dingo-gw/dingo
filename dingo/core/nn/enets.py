import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from nflows.nn.nets.resnet import ResidualBlock


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

    Layer specs
    --------
        input dimension:    (batch_size, num_blocks, num_channels, num_bins)
        output dimension:   (batch_size, 2 * n_rb * num_blocks)
    """

    def __init__(self,
                 input_dims,
                 n_rb,
                 V_rb_list,
                 ):
        """
        Parameters
        ----------
        input_dims : tuple
            dimensions of input batch, omitting batch dimension
            input_dims = (num_blocks, num_channels, num_bins)
        n_rb : int
            number of reduced basis elements used for projection
            the output dimension of the layer is 2 * n_rb * num_blocks
        V_rb_list : list of np.arrays
            list with V matrices of the reduced basis SVD projection,
            convention for SVD matrix decomposition: U @ s @ V^h
        """

        super(LinearProjectionRB, self).__init__()

        self.input_dims = input_dims
        if len(self.input_dims) != 3:
            raise ValueError('Exactly 3 axes required: blocks, channels, bins')
        self.num_blocks, self.num_channels, self.num_bins = self.input_dims
        self.n_rb = n_rb
        self.test_dimensions(V_rb_list)

        # define a linear projection layer for each block
        layers = []
        for ind in range(self.num_blocks):
            layers.append(
                nn.Linear(self.num_bins * self.num_channels, self.n_rb * 2))
        self.layers = nn.ModuleList(layers)

        # initialize layers with reduced basis
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
                'Number of channels needs to be at least 2, for real and '
                'imaginary parts.')
        if len(V_rb_list) != self.num_blocks:
            raise ValueError(
                'There must be exactly one reduced basis matrix V for each '
                'block.')
        for V in V_rb_list:
            if not isinstance(V, np.ndarray) or len(V.shape) != 2:
                raise ValueError(
                    'Reduced basis matrix V must be a numpy array with 2 axes.')
            if V.shape[0] != self.num_bins:
                raise ValueError(
                    'Number of input bins and number of rows in rb matrix V '
                    'need to match.')
            if V.shape[1] < self.n_rb:
                raise ValueError(
                    'More reduced basis elements requested than available.')

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
        for ind, layer in enumerate(self.layers):
            V = V_rb_list[ind]

            # truncate V to n_rb basis elements
            V = V[:, :n]
            V_real, V_imag = torch.from_numpy(V.real).float(), \
                             torch.from_numpy(V.imag).float()

            # initialize all weights and biases with zero
            layer.weight.data = torch.zeros_like(layer.weight.data)
            layer.bias.data = torch.zeros_like(layer.bias.data)

            # load matrix V into weights
            layer.weight.data[:n, :k] = torch.transpose(V_real, 1, 0)
            layer.weight.data[n:, :k] = torch.transpose(V_imag, 1, 0)
            layer.weight.data[:n, k:2 * k] = - torch.transpose(V_imag, 1, 0)
            layer.weight.data[n:, k:2 * k] = torch.transpose(V_real, 1, 0)

    def forward(self, x):
        assert x.shape[1:] == (
            self.num_blocks, self.num_channels, self.num_bins)
        out = []
        for ind in range(self.num_blocks):
            out.append(self.layers[ind](x[:, ind, ...].flatten(start_dim=1)))
        x = torch.cat(out, dim=1)
        return x


if __name__ == '__main__':
    pass
