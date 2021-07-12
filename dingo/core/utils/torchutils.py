import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Tuple


def get_activation_function_from_string(
        activation_name: str
):
    """
    Returns an activation function, based on the name provided.

    :param activation_name: str
        name of the activation function, one of {'elu', 'relu', 'leaky_rely'}
    :return: function
        corresponding activation function
    """
    if activation_name.lower() == 'elu':
        return F.elu
    elif activation_name.lower() == 'relu':
        return F.relu
    elif activation_name.lower() == 'leaky_relu':
        return F.leaky_relu
    else:
        raise ValueError('Invalid activation function.')


def forward_pass_with_unpacked_tuple(
        model: nn.Module,
        x: Union[Tuple, torch.Tensor],
):
    """
    Performs forward pass of model with input x. If x is a tuple, it return
    y = model(*x), else it returns y = model(x).
    :param model: nn.Module
        model for forward pass
    :param x: Union[Tuple, torch.Tensor]
        input for forward pass
    :return: torch.Tensor
        output of the forward pass, either model(*x) or model(x)
    """
    if isinstance(x, Tuple):
        return model(*x)
    else:
        return model(x)


def get_number_of_model_parameters(
        model: nn.Module,
        requires_grad_flags: tuple = (True, False),
):
    """
    Counts parameters of the module. The list requires_grad_flag can be used
    to specify whether all parameters should be counted, or only those with
    requires_grad = True or False.
    :param model: nn.Module
        model
    :param requires_grad_flags: tuple
        tuple of bools, for requested requires_grad flags
    :return:
        number of parameters of the model with requested required_grad flags
    """
    num_params = 0
    for p in list(model.parameters()):
        if p.requires_grad in requires_grad_flags:
            n = 1
            for s in list(p.size()):
                n = n * s
            num_params += n
    return num_params
