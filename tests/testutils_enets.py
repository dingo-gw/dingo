import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from dingo.core.utils.torchutils import forward_pass_with_unpacked_tuple


########################
# test utils for enet  #
########################


def generate_1d_series_data(batch_size, num_bins, alpha=20, beta=0):
    """Generate a set of batch_size 1d series with num_bins bins."""

    theta = np.random.rand(batch_size, 2)
    x = np.linspace(0, 1, num_bins)
    y = np.zeros((batch_size, num_bins))
    for i in range(batch_size):
        y[i, :] = np.cos(
            x * (0.1 + theta[i, 0]) * alpha + theta[i, 1]) * x ** beta
    y = y * np.exp(2j * np.pi * x ** 2)
    return y


def generate_1d_datasets_and_reduced_basis(batch_size=1000, num_bins=200,
                                           plot=False, n_rb=10):
    """
    Generate two sets of batch_size 1d series with num_bins bins,
    and generate a reduced basis for these.
    """

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
    """
    Turn multiple sets of 1d series into a batch for a nn by concatenation.
    Real and imaginary parts are separated into separate channels.
    """

    y_batch = torch.ones(
        (y_list[0].shape[0], len(y_list), num_channels, y_list[0].shape[1]))
    for idx, y in enumerate(y_list):
        y_batch[:, idx, 0, :] = torch.from_numpy(y.real).float()
        y_batch[:, idx, 1, :] = torch.from_numpy(y.imag).float()
    return y_batch


def project_onto_reduced_basis_and_concat(y_list, V_rb_list, n_rb):
    """
    Projects each y in y_list onto the corresponding reduced basis in V_rb_list.
    n_rb basis elements are used. For each y, the real and imaginary parts
    are concatenated. Finally, the reduced basis representations of all y are
    concatenated and returned.
    """

    projections = [[(y[:] @ V_rb[:, :n_rb]).real, (y[:] @ V_rb[:, :n_rb]).imag]
                   for y, V_rb in zip(y_list, V_rb_list)]
    projections = list(chain.from_iterable(projections))
    return np.concatenate(projections, axis=1)


def check_model_forward_pass(model, expected_output_shape, input_shape=None,
                             batch_size=100, x=None):
    """
    This function tests the forward pass of the model. It generates random
    input x with shape (batch_size, *input_shape) and performs a forward
    pass. The output y = model(x) is checked for the correct output shape,
    y.shape == (batch_size, *expected_output_shape), and whether distinct
    inputs generate distinct outputs.

    :param model: model to be checked
    :param input_shape: shape of the input data, omitting batch dimension
    :param expected_output_shape: expected shape of the output, omitting
    batch dimension
    :param batch_size: batch size
    :param x: input to model, if provided input is not generated
    """
    if x is None:
        assert input_shape is not None, \
            'input_shape required when x not provided.'
        x = torch.rand((batch_size, *input_shape))
    y = forward_pass_with_unpacked_tuple(model,x)  # replaces y = model(x)
    # check output shape
    assert y.shape[1:] == (*expected_output_shape,), \
        'Unexpected shape of model output.'
    # check that results are different for different inputs
    permuted_indices = [idx - 1 for idx in range(y.shape[0])]
    loss_fn = nn.L1Loss(reduction='none')
    loss = loss_fn(y, y[permuted_indices])
    assert torch.median(loss) / torch.median(torch.abs(y)) > 0.01, \
        'Model outputs could be insensitive to input. Check manually.'


def check_model_backward_pass(model, input_shape=None, batch_size=100, x=None):
    """
    This function tests the backward pass of the model and the optimizer
    step. It generates random input x with shape (batch_size, *input_shape),
    performs a forward pass y=model(x), computes the L1 loss between the output
    y and a random target vector with the same shape. This function then
    checks, whether the loss decreases after a backward pass and an optimizer
    step.

    :param model: model to be checked
    :param input_shape: shape of the input data, omitting batch dimension
    :param batch_size: batch size
    :param x: input to model, if provided input is not generated
    """
    if x is None:
        assert input_shape is not None, \
            'input_shape required when x not provided.'
        x = torch.rand((batch_size, *input_shape))
    y_0 = forward_pass_with_unpacked_tuple(model, x) # replaces y_0 = model(x)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.L1Loss()
    target = torch.rand_like(y_0)
    loss_before = loss_fn(y_0, target)
    loss_before.backward()
    optimizer.step()
    y_1 = forward_pass_with_unpacked_tuple(model, x) # replaces y_1 = model(x)
    loss_after = loss_fn(y_1, target)
    assert loss_after < loss_before, \
        'Loss does not decrease with optimizer step.'
    return y_0, y_1
