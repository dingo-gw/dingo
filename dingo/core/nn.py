print('Import successful!')

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from nflows.nn.nets.resnet import ResidualBlock

class EmbeddingNetworkFCNRes(nn.Module):
    """
    """
    def __init__(self,
                 input_dim,
                 Nrb,
                 input_channels,
                 input_blocks,
                 V_rb,
                 output_dim,
                 save_V = False,
                 res_hidden_dims = None,
                 res_activation = F.relu,
                 res_dropout_probability = 0.0,
                 res_batch_norm = False,
                 ):
        super(EmbeddingNetworkFCNRes, self).__init__()

        # Module 1
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.input_blocks = input_blocks
        self.Nrb = Nrb
        self.output_dim_single = self.Nrb * 2 # output dim for a single detector is twice the number of rb basis elements, for imag and real part
        self.output_dim = self.output_dim_single * self.input_blocks
        layers = []
        for ind in range(self.input_blocks):
            layers.append(nn.Linear(self.input_dim * input_channels, self.output_dim_single))
        self.layers = nn.ModuleList(layers)

        if V_rb is not None:
            self.init_layers(V_rb, save_V)


        # Module 2
        in_features = self.output_dim
        out_features = output_dim
        res_hidden_features = res_hidden_dims
        res_use_batch_norm = res_batch_norm
        self.res_hidden_features = res_hidden_features
        self.initial_layer = nn.Linear(in_features, res_hidden_features[0])
        num_blocks = len(res_hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=res_hidden_features[n],
                    context_features=None,
                    activation=res_activation,
                    dropout_probability=res_dropout_probability,
                    use_batch_norm=res_use_batch_norm,
                )
                for n in range(num_blocks)
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.Linear(res_hidden_features[n - 1], res_hidden_features[n])
                if res_hidden_features[n - 1] != res_hidden_features[n]
                else nn.Identity()
                for n in range(1, num_blocks)
            ]
        )
        self.final_layer = nn.Linear(res_hidden_features[-1], out_features)

        print('Module 1: {:} -> {:}'.format(self.input_dim * self.input_channels * self.input_blocks, self.output_dim))
        print('Module 2: {:} -> {:}'.format(in_features, out_features))





    def init_layers(self, V_path, save_V):
        stds = torch.sqrt(torch.ones((self.input_blocks, self.Nrb), dtype=torch.float32, device='cpu'))
        # There must either be a single reduced basis, that is used for all detectors, or one each
        if len(V_path) == 1:
            ind_V = [0] * len(self.layers)
        else:
            assert len(V_path) == len(self.layers), 'There must either be one reduced basis used for all detectors, or one for each.'
            ind_V = np.arange(len(V_path))
        for ind,layer in enumerate(self.layers):
            # load V
            V = np.load(V_path[ind_V[ind]])
            print('Initializing first layer for detector {:} with reduced basis {:}.'.format(ind, V_path[ind_V[ind]].split('/')[-1]))
            # truncate V to only contain relevant frequencies and Nrb basis elements
            V = V[V.shape[0] - self.input_dim:,:self.Nrb]
            Vr = torch.from_numpy(V.real).float()
            Vi = torch.from_numpy(V.imag).float()
            # initialize weights and biases with zero
            layer.weight.data = torch.zeros_like(layer.weight.data)
            layer.bias.data = torch.zeros_like(layer.bias.data)
            # load matrices V into weights
            layer.weight.data[:self.Nrb,:self.input_dim] = (Vr*stds[ind]).permute(1,0)
            layer.weight.data[self.Nrb:,:self.input_dim] = (Vi*stds[ind]).permute(1,0)
            layer.weight.data[:self.Nrb,self.input_dim:2*self.input_dim] = - (Vi*stds[ind]).permute(1,0)
            layer.weight.data[self.Nrb:,self.input_dim:2*self.input_dim] = (Vr*stds[ind]).permute(1,0)
        if save_V:
            self.V = V


    def forward(self, x):

        # Module 1
        bs, d, c, f = x.shape
        if f == self.input_dim + 1:
            flag = x[0,0,1,0]
            detector_times = x[:,:,0,0]
            x = x[...,1:]
        assert d == self.input_blocks
        assert c == self.input_channels

        out = []
        for ind in range(d):
            out.append(self.layers[ind](x[:,ind,...].flatten(start_dim=1)))

        x = torch.cat(out, dim=1)


        # Module 2
        temps = self.initial_layer(x)
        for n, block in enumerate(self.blocks):
            temps = block(temps, context=None)
            if n < (len(self.blocks) - 1):
                temps = self.resize_layers[n](temps)
        x = self.final_layer(temps)

        if f == self.input_dim + 1:
            if flag == 1:
                x = torch.cat((x, detector_times), axis=1)
            else:
                x = torch.cat((x, detector_times[:,1:]), axis=1)

        return x


def create_embedding_net_fcn_res(**kwargs):

    if kwargs['res_activation'] == 'elu':
        kwargs['res_activation'] = F.elu
    elif kwargs['res_activation'] == 'relu':
        kwargs['res_activation'] = F.relu
    elif kwargs['res_activation'] == 'leaky_relu':
        kwargs['res_activation'] = F.leaky_relu
    else:
        kwargs['res_activation'] = F.relu   # Default
        print('Invalid activation function specified. Using ReLU.')

    return EmbeddingNetworkFCNRes(**kwargs)


