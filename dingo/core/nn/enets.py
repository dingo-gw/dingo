import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from nflows.nn.nets.resnet import ResidualBlock

class LinearProjectionRB(nn.Module):
    """
    A compression layer that reduces the input dimensionality by projection onto a reduced basis.
    The input data is of shape (batch_size, num_blocks, num_channels, num_bins).
    Each block is treated independently. For GW use case, a block corresponds to a detector.
    Each block has num_channels >= 2 channels. Channel 0 and 1 represent the real and imaginary part, respectively.
    Each channel has num_bins bins.
    """

    def __init__(self,
                 input_dims,
                 n_rb,
                 V_rb,
                 ):
        """
        Parameters
        ----------
        input_dims : tuple
            dimensions of input batch, omitting batch dimension
            input_dims = (num_blocks, num_channels, num_bins)
        n_rb : int
            number of reduced basis elements used for projection
            the output dimension is 2 * n_rb * num_blocks
        V_rb : list of np.arrays
            list with v matrices of the reduced basis SVD projection
        """

        super(LinearProjectionRB, self).__init__()

        self.test_dimensions(input_dims, n_rb, V_rb)
        self.input_dims = input_dims
        self.num_blocks, self.num_channels, self.num_bins = self.input_dims
        self.n_rb = n_rb

        # define a linear projection layer for each block
        layers = []
        for ind in range(self.num_blocks):
            layers.append(nn.Linear(self.num_bins * self.num_channels, self.n_rb * 2))
        self.layers = nn.ModuleList(layers)
        # initialize layers with reduced basis
        self.init_layers(V_rb)

    @property
    def input_dim(self):
        return self.num_bins * self.num_channels * self.num_blocks

    @property
    def output_dim(self):
        return 2 * self.n_rb * self.num_blocks

    def test_dimensions(self, input_dims, n_rb, V_rb):
        assert len(input_dims) == 3, 'Exactly 3 axes required: blocks, channels, bins'
        assert input_dims[1] >= 2, 'Number of channels needs to be at least 2, for real and imaginary parts.'
        assert len(V_rb) == input_dims[0], 'There must be exactly one reduced basis matrix v for each block.'
        for v in V_rb:
            assert v.shape[0] == input_dims[2], 'Number of input bins and number of rows in V_rb need to match.'
            assert v.shape[1] >= n_rb, 'More reduced basis elements requested than available.'


    def init_layers(self, V_rb):
        # loop through layers and initialize them individually
        for ind,layer in enumerate(self.layers):
            v = V_rb[ind]
            # truncate v to n_rb basis elements
            v = v[:,:self.n_rb]
            v_real, v_imag = torch.from_numpy(v.real).float(), torch.from_numpy(v.imag).float()
            # initialize all weights and biases with zero
            layer.weight.data = torch.zeros_like(layer.weight.data)
            layer.bias.data = torch.zeros_like(layer.bias.data)
            # load matrix v into weights
            layer.weight.data[:self.n_rb,:self.num_bins] = (v_real).permute(1,0)
            layer.weight.data[self.n_rb:,:self.num_bins] = (v_imag).permute(1,0)
            layer.weight.data[:self.n_rb,self.num_bins:2*self.num_bins] = - (v_imag).permute(1,0)
            layer.weight.data[self.n_rb:,self.num_bins:2*self.num_bins] = (v_real).permute(1,0)


    def forward(self, x):
        assert x.shape[1:] == (self.num_blocks, self.num_channels, self.num_bins)
        out = []
        for ind in range(self.num_blocks):
            out.append(self.layers[ind](x[:,ind,...].flatten(start_dim=1)))
        x = torch.cat(out, dim=1)
        return x




if __name__ == '__main__':
    pass