from torch.nn import functional as F


def get_activation_function_from_string(activation_name):
    if activation_name.lower() == 'elu':
        return F.elu
    elif activation_name.lower() == 'relu':
        return F.relu
    else:
        raise ValueError('Invalid activation function.')
