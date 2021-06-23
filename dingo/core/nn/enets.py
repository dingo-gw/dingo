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
                 v_rb,
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
        v_rb : list of np.arrays
            list with v matrices of the reduced basis SVD projection
        """

        super(LinearProjectionRB, self).__init__()

        self.test_dimensions(input_dims, n_rb, v_rb)
        self.input_dims = input_dims
        self.num_blocks, self.num_channels, self.num_bins = self.input_dims
        self.n_rb = n_rb

        # self.output_dim_single = self.n_rb * 2 # output dim for a single detector is twice the number of rb basis elements, for imag and real part
        # self.output_dim = self.output_dim_single * self.input_blocks

        # define a linear projection layer for each block
        layers = []
        for ind in range(self.num_blocks):
            layers.append(nn.Linear(self.num_bins * self.num_channels, self.n_rb * 2))
        self.layers = nn.ModuleList(layers)

        # initialize layers with
        self.init_layers(v_rb)

        print('Module 1: {:} -> {:}'.format(self.input_dim, self.output_dim))

    @property
    def input_dim(self):
        return self.num_bins * self.num_channels * self.num_blocks

    @property
    def output_dim(self):
        return 2 * self.n_rb * self.num_blocks

    def test_dimensions(self, input_dims, n_rb, v_rb):
        assert len(input_dims) == 3, 'Exactly 3 axes required: blocks, channels, bins'
        assert input_dims[1] >= 2, 'Number of channels needs to be at least 2, for real and imaginary parts.'
        assert len(v_rb) == input_dims[0], 'There must be exactly one reduced basis matrix v for each block.'
        for v in v_rb:
            assert v.shape[0] == input_dims[2], 'Number of input bins and number of rows in v_rb need to match.'
            assert v.shape[1] >= n_rb, 'More reduced basis elements requested than available.'


    def init_layers(self, v_rb):
        # loop through layers and initialize them individually
        for ind,layer in enumerate(self.layers):
            print('Initializing reduced basis projection layer for block {:}.'.format(ind))
            v = v_rb[ind]
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
    import matplotlib.pyplot as plt
    from scipy import linalg

    num_bins = 200
    n = 1000
    n_rb = 10
    x = np.linspace(0,1,num_bins)
    theta = np.random.rand(n, 2)
    y1, y2 = np.zeros((n, num_bins)), np.zeros((n, num_bins))
    for i in range(n):
        y1[i,:] = np.cos(x*(0.1+theta[i,0])*20 + theta[i,1])
        y2[i,:] = np.cos(x*(0.1+theta[i,0])*20 + theta[i,1]) * x
    y1 = y1 * np.exp(2j*np.pi*x**2)
    y2 = y2 * np.exp(2j*np.pi*x**2)

    _, _, Vh1 = linalg.svd(y1)
    V1 = Vh1.T.conj()
    _, _, Vh2 = linalg.svd(y2)
    V2 = Vh2.T.conj()

    plt.plot(x, y1[0].real)
    plt.plot(x, (y1[0]@V1[:,:n_rb]@Vh1[:n_rb,:]).real)
    plt.plot(x, (y1[0]@V2[:,:n_rb]@Vh2[:n_rb,:]).real)
    plt.show()

    plt.plot(x, (y1[0]@V1[:,:n_rb]@Vh1[:n_rb,:]).real - y1[0].real)
    plt.plot(x, (y1[0]@V2[:,:n_rb]@Vh2[:n_rb,:]).real - y1[0].real)
    plt.show()

    plt.plot(x, (y2[0]@V1[:,:n_rb]@Vh1[:n_rb,:]).real - y2[0].real)
    plt.plot(x, (y2[0]@V2[:,:n_rb]@Vh2[:n_rb,:]).real - y2[0].real)
    plt.show()

    layer1 = LinearProjectionRB(input_dims=(2,3,num_bins), n_rb=10, v_rb=[V1, V2])

    y_a = torch.zeros((n, 2, 3, num_bins))
    y_a[:,0,0,:] = torch.from_numpy(y1.real).float()
    y_a[:,0,1,:] = torch.from_numpy(y1.imag).float()
    y_a[:,1,0,:] = torch.from_numpy(y2.real).float()
    y_a[:,1,1,:] = torch.from_numpy(y2.imag).float()

    y_tmp = np.ones_like(y1)
    y_b = torch.zeros((n, 2, 3, num_bins))
    y_b[:,0,0,:] = torch.from_numpy(y_tmp.real).float()
    y_b[:,0,1,:] = torch.from_numpy(y_tmp.imag).float()
    y_b[:,1,0,:] = torch.from_numpy(y_tmp.real).float()
    y_b[:,1,1,:] = torch.from_numpy(y_tmp.imag).float()

    out_a = np.array(layer1(y_a).detach())
    out_b = np.array(layer1(torch.ones_like(y_a)).detach())

    ref_a = np.concatenate(((y1[:]@V1[:,:n_rb]).real,
                            (y1[:]@V1[:,:n_rb]).imag,
                            (y2[:]@V1[:,:n_rb]).real,
                            (y2[:]@V1[:,:n_rb]).imag), axis=1)
    ref_b = np.concatenate(((y_tmp[:]@V1[:,:n_rb]).real,
                            (y_tmp[:]@V1[:,:n_rb]).imag,
                            (y_tmp[:]@V1[:,:n_rb]).real,
                            (y_tmp[:]@V1[:,:n_rb]).imag), axis=1)

    print('done')