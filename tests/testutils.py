import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import torch



########################
# test utils for enet  #
########################


def generate_1d_series_data(batch_size, num_bins, alpha=20, beta=0):
    '''Generate a set of batch_size 1d series with num_bins bins.'''
    theta = np.random.rand(batch_size, 2)
    x = np.linspace(0,1,num_bins)
    y = np.zeros((batch_size, num_bins))
    for i in range(batch_size):
        y[i,:] = np.cos(x*(0.1+theta[i,0])*alpha + theta[i,1]) * x**beta
    y = y * np.exp(2j * np.pi * x**2)
    return y


def generate_1d_datasets_and_reduced_basis(batch_size=1000, num_bins=200, plot=False):
    '''
    Generate two sets of batch_size 1d series with num_bins bins,
    and generate a reduced basis for these.
    '''

    # generate datasets
    y1 = generate_1d_series_data(batch_size, num_bins, alpha=20, beta=0)
    y2 = generate_1d_series_data(batch_size, num_bins, alpha=20, beta=1)
    # generate reduced basis
    _, _, Vh1 = linalg.svd(y1)
    V1 = Vh1.T.conj()
    _, _, Vh2 = linalg.svd(y2)
    V2 = Vh2.T.conj()
    # plot data
    if plot:
        plt.plot(y1[0].real)
        plt.plot((y1[0] @ V1[:, :n_rb] @ Vh1[:n_rb, :]).real)
        plt.plot((y1[0] @ V2[:, :n_rb] @ Vh2[:n_rb, :]).real)
        plt.show()
        plt.plot((y1[0] @ V1[:, :n_rb] @ Vh1[:n_rb, :]).real - y1[0].real)
        plt.plot((y1[0] @ V2[:, :n_rb] @ Vh2[:n_rb, :]).real - y1[0].real)
        plt.show()
        plt.plot((y2[0] @ V1[:, :n_rb] @ Vh1[:n_rb, :]).real - y2[0].real)
        plt.plot((y2[0] @ V2[:, :n_rb] @ Vh2[:n_rb, :]).real - y2[0].real)
        plt.show()
    return (y1, y2), (V1, V2)


def get_y_batch(y_list, num_channels=3):
    '''
    Turn multiple sets of 1d series into a batch for a nn by concatenation.
    Real and imaginary parts are separated into separate channels.
    '''
    y_batch = torch.ones((y_list[0].shape[0], len(y_list), num_channels, y_list[0].shape[1]))
    for idx, y in enumerate(y_list):
        y_batch[:,idx,0,:] = torch.from_numpy(y.real).float()
        y_batch[:,idx,1,:] = torch.from_numpy(y.imag).float()
    return y_batch


def project_onto_reduced_basis_and_concat(y_list, V_rb_list, n_rb):
    '''
    Projects each y in y_list onto the corresponding reduced basis in V_rb_list.
    n_rb basis elements are used.
    For each y, the real and imaginary parts are concatenated.
    Finally, the reduced basis representations of all y are concatenated and returned.
    '''
    projections = [[(y[:] @ V_rb[:, :n_rb]).real, (y[:] @ V_rb[:, :n_rb]).imag] for y, V_rb in zip(y_list, V_rb_list)]
    projections = [item for sublist in projections for item in sublist]
    return np.concatenate(projections, axis=1)