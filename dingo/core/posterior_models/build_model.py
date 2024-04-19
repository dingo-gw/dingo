import torch

from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet
from dingo.core.nn.transformer import create_transformer_enet
from dingo.core.posterior_models.flow_matching import FlowMatching
from dingo.core.posterior_models.normalizing_flow import NormalizingFlow
from dingo.core.posterior_models.pretraining_model import PretrainingModel
from dingo.core.posterior_models.score_matching import ScoreDiffusion


def build_model_from_kwargs(
    filename: str = None,
    settings: dict = None,
    pretraining: bool = False,
    pretrained_embedding_net: torch.nn.Module = None,
    **kwargs,
):
    """
    Returns the built model (from settings file or rebuild from file). Extracts the relevant arguments (normalizing flow
    or continuous flow) from setting.

    Parameters
    ----------
    filename: str=None
        filename of the model to load
    settings: dict=None
        settings file for building a new model
    pretraining: bool=False
        whether to use the pretrained embedding network
    pretrained_embedding_net: torch.nn.Module=None
        pretrained embedding network
    kwargs

    Returns
    -------
    PosteriorModel
    """
    # either filename or settings must be provided
    assert filename is not None or settings is not None

    models_dict = {
        "normalizing_flow": NormalizingFlow,
        "flow_matching": FlowMatching,
        "score_matching": ScoreDiffusion,
        "pretraining": PretrainingModel,
    }
    embedding_net_builder_dict = {
        "denseresidualnet": create_enet_with_projection_layer_and_dense_resnet,
        "transformer": create_transformer_enet,
        "no_embedding": None,
    }

    if filename is not None:
        if "device" in kwargs:
            device = kwargs["device"]
        else:
            device = "cpu"
        d = torch.load(filename, map_location=device)
        posterior_model_type = d["metadata"]["train_settings"]["model"][
            "posterior_model_type"
        ]
        if "embedding_kwargs" in d["metadata"]["train_settings"]["model"].keys():
            embedding_network_type = d["metadata"]["train_settings"]["model"][
                "embedding_type"
            ]
        else:
            embedding_network_type = "no_embedding"
    else:
        posterior_model_type = settings["train_settings"]["model"][
            "posterior_model_type"
        ]
        if "embedding_kwargs" in settings["train_settings"]["model"].keys():
            embedding_network_type = settings["train_settings"]["model"][
                "embedding_type"
            ]
        else:
            embedding_network_type = "no_embedding"

    if not posterior_model_type.lower() in models_dict:
        raise ValueError("No valid posterior model specified.")
    if not embedding_network_type.lower() in embedding_net_builder_dict:
        raise ValueError("No valid embedding network type specified.")
    if (
        posterior_model_type.lower() != "normalizing_flow"
        and embedding_network_type.lower() == "no_embedding"
    ):
        raise ValueError(
            f"No embedding_kwargs were specified. It is not possible to train an unconditional"
            f"posterior model for {posterior_model_type}. If you want to train an unconditional posterior"
            f"model, use normalizing_flow instead."
        )

    model = models_dict[posterior_model_type.lower()]
    emb_net_builder = embedding_net_builder_dict[embedding_network_type.lower()]
    if settings is not None:
        if posterior_model_type.lower() == "normalizing_flow":
            settings["train_settings"]["model"]["posterior_kwargs"].update(
                settings["train_settings"]["model"].get("nf_kwargs", {})
            )

        if posterior_model_type.lower() in ["flow_matching", "score_matching"]:
            settings["train_settings"]["model"]["posterior_kwargs"].update(
                settings["train_settings"]["model"].get("cf_kwargs", {})
            )

        # if posterior_model_type.lower() in ["flow_matching", "score_matching"]:
        if "nf_kwargs" in settings["train_settings"]["model"].keys():
            del settings["train_settings"]["model"]["nf_kwargs"]
        # if posterior_model_type.lower() == "normalizing_flow":
        if "cf_kwargs" in settings["train_settings"]["model"].keys():
            del settings["train_settings"]["model"]["cf_kwargs"]

    pm = model(
        model_filename=filename,
        metadata=settings,
        embedding_net_builder=emb_net_builder,
        **kwargs,
    )

    if not pretraining and pretrained_embedding_net is not None:
        pm.network.embedding_net = pretrained_embedding_net

    return pm


def autocomplete_model_kwargs(
    model_kwargs: dict,
    data_sample: dict = None,
    input_dim: int = None,
    context_dim: int = None,
):
    """
    Autocomplete the model kwargs from train_settings and data_sample from
    the dataloader:
    (*) set input dimension of embedding net to shape of data_sample[1]
    (*) set dimension of parameter space of posterior model to len(data_sample[0])
    (*) not in this branch: set added_context flag of embedding net if required for gnpe proxies
        -> not possible to use gnpe with transformer because no quick-fix solution available
    (*) set context dim of posterior model to output dim of embedding net + gnpe proxy dim
    (*) if pretraining: set input dimension of pretraining net to output dimension of embedding network and
                        set output dimension of pretraining net to number of posterior parameters

    Parameters
    ----------
    model_kwargs: dict
        keyword arguments of model
    data_sample: list=None
        Sample from dataloader (e.g., wfd[0]) used for autocompletion.
        Should be of format [parameters, GW data, gnpe_proxies], where the
        last element is only there is gnpe proxies are required.
    input_dim: int=None
        dimension of input
    context_dim: int=None
        dimension of context

    Returns
    ----------
    model_kwargs: dict
        updated, autocompleted model_kwargs
    """

    # If provided, extract settings from the data sample. Otherwise, use provided kwargs. Since input_dim always needs
    # to be provided, we can use this to verify that they are mutually exclusive.
    assert (
        data_sample is not None
        and input_dim is None
        or data_sample is None
        and input_dim is not None
    )

    if data_sample is not None:
        # set input dims from ifo_list and domain information
        if model_kwargs["embedding_type"].lower() == "denseresidualnet":
            model_kwargs["embedding_kwargs"]["input_dims"] = list(data_sample[1].shape)
            context_dim = model_kwargs["embedding_kwargs"]["output_dim"]
        elif model_kwargs["embedding_type"].lower() == "transformer":
            if "tokenizer_kwargs" in model_kwargs["embedding_kwargs"]:
                model_kwargs["embedding_kwargs"]["tokenizer_kwargs"]["input_dims"] = list(
                    data_sample[1].shape
                )
                model_kwargs["embedding_kwargs"]["tokenizer_kwargs"][
                    "output_dim"
                ] = model_kwargs["embedding_kwargs"]["transformer_kwargs"]["d_model"]
            if "final_net_kwargs" in model_kwargs["embedding_kwargs"]:
                model_kwargs["embedding_kwargs"]["final_net_kwargs"][
                    "input_dim"
                ] = model_kwargs["embedding_kwargs"]["transformer_kwargs"]["d_model"]
                context_dim = model_kwargs["embedding_kwargs"]["final_net_kwargs"][
                    "output_dim"
                ]
            else:
                context_dim = model_kwargs["embedding_kwargs"]["transformer_kwargs"]["d_model"]

        elif model_kwargs["embedding_type"] == "no_embedding":
            context_dim = None
            print("No embedding network specified.")
        else:
            raise ValueError(
                f"Embedding type {model_kwargs['embedding_type']} not in [DenseResidualNet, transformer, "
                f"no_embedding]"
            )

        # check for pretraining
        if model_kwargs["posterior_model_type"] == "pretraining":
            # set input and output dim for pretraining network
            model_kwargs["posterior_kwargs"]["input_dim"] = context_dim
            model_kwargs["posterior_kwargs"]["output_dim"] = len(data_sample[0])
        else:
            # set dimension of parameter space of posterior model
            model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])

            # additional information for GNPE
            # TODO: get GNPE running with transformer for backwards compatibility
            # Currently: not possible to use GNPE, no quick fix available on how to determine whether gnpe is used
            # previously: ```try: len(data_sample[2])```
            # but position element of transformer input has values at data_sample[2] ...
            # if len(data_sample) == 3:
            #    # set added_context flag of embedding net if gnpe proxies are required
            #    # set context dim of nsf to output dim of embedding net + gnpe proxy dim
            #    gnpe_proxy_dim = len(data_sample[2])
            #    model_kwargs["embedding_kwargs"]["added_context"] = True
            #    model_kwargs["posterior_kwargs"]["context_dim"] = (
            #        context_dim + gnpe_proxy_dim
            #    )
            # else:
            #    model_kwargs["embedding_kwargs"]["added_context"] = False
            #    model_kwargs["posterior_kwargs"]["context_dim"] = context_dim
            model_kwargs["posterior_kwargs"]["context_dim"] = context_dim
    else:
        model_kwargs["posterior_kwargs"]["input_dim"] = input_dim
        model_kwargs["posterior_kwargs"]["context_dim"] = context_dim

    # return model_kwargs
