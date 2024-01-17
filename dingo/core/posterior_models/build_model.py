import torch

from dingo.core.posterior_models.normalizing_flow import NormalizingFlow
from dingo.core.posterior_models.flow_matching import FlowMatching
from dingo.core.posterior_models.score_matching import ScoreDiffusion


# TODO: where to put this to avoid cyclic imports?
def build_model_from_kwargs(filename=None, settings=None, **kwargs):
    """
    Returns the built model (from settings file or rebuild from file). Extracts the relevant arguments (normalizing flow
    or continuous flow) from setting.

    Parameters
    ----------
    filename
    settings
    kwargs

    Returns
    -------

    """
    # either filename or settings must be provided
    assert filename is not None or settings is not None

    models_dict = {
        "normalizing_flow": NormalizingFlow,
        "flow_matching": FlowMatching,
        "score_matching": ScoreDiffusion,
    }

    if filename is not None:
        d = torch.load(filename, map_location="cpu")
        type = d["metadata"]["train_settings"]["model"]["type"]
    else:
        type = settings["train_settings"]["model"]["type"]

    if not type.lower() in models_dict:
        raise ValueError("No valid posterior model specified.")

    model = models_dict[type.lower()]
    # TODO copy the relevant posterior model arguments to posterior model and delete
    # TODO non-necessary arguments currently keeps all arguments for flow_matching and
    # TODO score_matching, e.g., sigma_min for score_matching
    # delete and or modify parameter file.
    if settings is not None:
        if type.lower() == "normalizing_flow":
            settings["train_settings"]["model"]["posterior_kwargs"].update(
                settings["train_settings"]["model"].get("nf_kwargs", {})
            )

        if type.lower() in ["flow_matching", "score_matching"]:
            settings["train_settings"]["model"]["posterior_kwargs"].update(
                settings["train_settings"]["model"].get("cf_kwargs", {})
            )

        # if type.lower() in ["flow_matching", "score_matching"]:
        if "nf_kwargs" in settings["train_settings"]["model"].keys():
            del settings["train_settings"]["model"]["nf_kwargs"]
        # if type.lower() == "normalizing_flow":
        if "cf_kwargs" in settings["train_settings"]["model"].keys():
            del settings["train_settings"]["model"]["cf_kwargs"]

    return model(model_filename=filename, metadata=settings, **kwargs)


def autocomplete_model_kwargs(
    model_kwargs, data_sample=None, input_dim=None, context_dim=None
):
    """
    Autocomplete the model kwargs from train_settings and data_sample from
    the dataloader:
    (*) set input dimension of embedding net to shape of data_sample[1]
    (*) set dimension of nsf parameter space to len(data_sample[0])
    (*) set added_context flag of embedding net if required for gnpe proxies
    (*) set context dim of nsf to output dim of embedding net + gnpe proxy dim

    :param train_settings: dict
        train settings as loaded from .yaml file
    :param data_sample: list
        Sample from dataloader (e.g., wfd[0]) used for autocomplection.
        Should be of format [parameters, GW data, gnpe_proxies], where the
        last element is only there is gnpe proxies are required.
    :return: model_kwargs: dict
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
        model_kwargs["embedding_kwargs"]["input_dims"] = list(data_sample[1].shape)
        # set dimension of parameter space of nsf
        model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])
        # set added_context flag of embedding net if gnpe proxies are required
        # set context dim of nsf to output dim of embedding net + gnpe proxy dim
        try:
            gnpe_proxy_dim = len(data_sample[2])
            model_kwargs["embedding_kwargs"]["added_context"] = True
            model_kwargs["posterior_kwargs"]["context_dim"] = (
                model_kwargs["embedding_kwargs"]["output_dim"] + gnpe_proxy_dim
            )
        except IndexError:
            model_kwargs["embedding_kwargs"]["added_context"] = False
            model_kwargs["posterior_kwargs"]["context_dim"] = model_kwargs[
                "embedding_kwargs"
            ]["output_dim"]
    else:
        model_kwargs["posterior_kwargs"]["input_dim"] = input_dim
        model_kwargs["posterior_kwargs"]["context_dim"] = context_dim

    # return model_kwargs
