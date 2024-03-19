import copy
from torch import nn

from dingo.core.utils import torchutils
from dingo.core.nn.enets import create_transformer_enet
from dingo.core.nn.resnet import DenseResidualNet


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

    def forward(self, x):
        x = torchutils.forward_pass_with_unpacked_tuple(self.embedding_net, x)
        return self.pretraining_net(x)


def create_transformer_embedding_with_resnet(
    embedding_net_kwargs: dict,
    pretraining_net_kwargs: dict,
):
    """
    Builds a transformer encoder embedding network and a ResNet 'decoder'.

    Parameters
    ----------
    embedding_net_kwargs : dict
        kwargs for embedding network
    pretraining_net_kwargs : dict
        kwargs for network used only in pretraining

    Returns
    -------
    nn.Module
    """

    embedding_net_kwargs = copy.deepcopy(embedding_net_kwargs)
    pretraining_net_kwargs = copy.deepcopy(pretraining_net_kwargs)

    embedding_net = create_transformer_enet(**embedding_net_kwargs)
    pretraining_net = None
    if "resnet" in pretraining_net_kwargs["type"]:
        pretraining_net_kwargs.pop("type")
        activation_fn = torchutils.get_activation_function_from_string(pretraining_net_kwargs["activation"])
        pretraining_net_kwargs["activation"] = activation_fn
        pretraining_net = DenseResidualNet(**pretraining_net_kwargs)
    else:
        ValueError(f"Type of pretraining network is {pretraining_net_kwargs['type']} which does not contain the options"
                   f"resnet.")
    model = ModelWrapper(embedding_net, pretraining_net)

    return model
