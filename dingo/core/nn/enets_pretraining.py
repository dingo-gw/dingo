import copy
from typing import Callable

from torch import nn

from dingo.core.nn.resnet import DenseResidualNet
from dingo.core.utils import torchutils


class ModelWrapper(nn.Module):
    """
    This class wraps any type of embedding-compression like architecture. It is
    required for multiple reasons. (i) some embedding networks take tuples as
    input, which is not supported generally. (ii) parallelization across multiple
    GPUs requires a forward method, but the relevant method for training
    might be different.
    """

    def __init__(self, embedding_net: nn.Module, pretraining_net: nn.Module):
        """
        Parameters
        ----------
        embedding_net: nn.Module
            Embedding network that maps some higher dimensional input to a
            lower dimension (usually the context dimension of the posterior model)
        pretraining_net: nn.Module
            Network that is only used during the pretraining stage
        """
        super(ModelWrapper, self).__init__()
        self.embedding_net = embedding_net
        self.pretraining_net = pretraining_net

    def forward(self, *x):
        x = torchutils.forward_pass_with_unpacked_tuple(self.embedding_net, *x)
        return self.pretraining_net(x)


def create_embedding_with_resnet(
    embedding_net_builder: Callable,
    embedding_kwargs: dict,
    posterior_kwargs: dict,
    initial_weights: dict = None,
):
    """
    Builds a transformer encoder embedding network and a ResNet 'decoder'.

    Parameters
    ----------
    embedding_net_builder: Callable
        function that builds embedding network
    embedding_kwargs : dict
        kwargs for embedding network
    posterior_kwargs : dict
        kwargs for network used only in pretraining
    initial_weights : dict
        Dictionary containing the initial weights for the SVD projection of the resnet embedding.
        This should have one key 'V_rb_list', with value a list of SVD V matrices (one for each
        detector).

    Returns
    -------
    nn.Module
    """
    embedding_kwargs = copy.deepcopy(embedding_kwargs)
    if initial_weights is not None:
        embedding_kwargs["V_rb_list"] = initial_weights["V_rb_list"]
    embedding_net = embedding_net_builder(**embedding_kwargs)

    pretraining_net_kwargs = copy.deepcopy(posterior_kwargs)
    if pretraining_net_kwargs["type"].lower() == "denseresidualnet":
        pretraining_net_kwargs.pop("type")
        pretraining_net_kwargs.pop("loss_function")
        activation_fn = torchutils.get_activation_function_from_string(
            pretraining_net_kwargs["activation"]
        )
        pretraining_net_kwargs["activation"] = activation_fn
        pretraining_net = DenseResidualNet(**pretraining_net_kwargs)
    else:
        raise ValueError(
            f"Type of pretraining network is {pretraining_net_kwargs['type']} which does not contain the "
            f"options resnet."
        )
    model = ModelWrapper(embedding_net, pretraining_net)

    return model
