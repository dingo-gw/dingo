from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.posterior_models.flow_matching import FlowMatchingPosteriorModel
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.score_matching import ScoreDiffusionPosteriorModel
from dingo.core.utils.backward_compatibility import (
    torch_load_with_fallback,
    update_data_config,
    update_model_config,
    check_minimum_version,
)


def build_model_from_kwargs(
    filename: str = None, settings: dict = None, **kwargs
) -> BasePosteriorModel:
    """
    Returns a PosteriorModel based on a saved network or settings dict.

    The function is careful to choose the appropriate PosteriorModel class (e.g.,
    for a normalizing flow, flow matching, or score matching).

    Parameters
    ----------
    filename: str
        Path to a saved network (.pt).
    settings: dict
        Settings dictionary.
    kwargs
        Arguments forwarded to the model constructor.

    Returns
    -------
    PosteriorModel
    """
    if (filename is None) == (settings is None):
        raise ValueError(
            "Either a filename or a settings dict must be provided, but not both."
        )

    models_dict = {
        "normalizing_flow": NormalizingFlowPosteriorModel,
        "flow_matching": FlowMatchingPosteriorModel,
        "score_matching": ScoreDiffusionPosteriorModel,
    }

    if filename is not None:
        d, _ = torch_load_with_fallback(filename, preferred_map_location="meta")
        if "version" in d:
            check_minimum_version(d["version"])
        else:
            # version was introduced in v0.3.3
            check_minimum_version("dingo=0.3.2")
        update_model_config(d["metadata"]["train_settings"]["model"])  # Backward compat
        posterior_model_type = d["metadata"]["train_settings"]["model"][
            "posterior_model_type"
        ]
    else:
        update_model_config(settings["train_settings"]["model"])  # Backward compat
        update_data_config(settings)
        posterior_model_type = settings["train_settings"]["model"][
            "posterior_model_type"
        ]

    if not posterior_model_type.lower() in models_dict:
        raise ValueError("No valid posterior model type specified.")

    model = models_dict[posterior_model_type.lower()]

    return model(model_filename=filename, metadata=settings, **kwargs)


def autocomplete_model_kwargs(model_kwargs: dict, data_sample: list):
    """
    Autocomplete the model kwargs from train_settings and data_sample from the dataloader:

    * set input dimension of embedding net to shape of data_sample[1]
    * set dimension of parameter space to len(data_sample[0])
    * set the number of context parameters (e.g., GNPE proxies), which the posterior
      network concatenates to the embedded data
    * set context dim of posterior model to output dim of embedding net + number of
      context parameters

    Parameters
    ----------
    model_kwargs: dict
        Model settings, which are modified in-place.
    data_sample: list
        Sample from dataloader (e.g., wfd[0]) used for autocomplection.
        Should be of format [parameters, *data, context_parameters], where data is
        [GW data] (resnet) or [GW data, position, token_mask] (transformer), and the
        last element is only there if the model has context parameters.
    """
    model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])

    embedding_type = (model_kwargs.get("embedding_type") or "resnet").lower()
    num_data_inputs = 3 if embedding_type == "transformer" else 1
    context_parameters = data_sample[1 + num_data_inputs :]
    num_context_parameters = len(context_parameters[0]) if context_parameters else 0
    # Only recorded when nonzero, so that networks without context parameters stay
    # loadable by earlier Dingo versions.
    if num_context_parameters:
        model_kwargs["num_context_parameters"] = num_context_parameters

    if embedding_type == "transformer":
        tokenizer_kwargs = model_kwargs["embedding_kwargs"]["tokenizer_kwargs"]
        tokenizer_kwargs["input_dim"] = int(data_sample[1].shape[-1])
        # Position layout: position_continuous_dim continuous columns, then one
        # column per categorical feature. Default: the last column is the single
        # categorical feature (GW: the detector's position in the training detector
        # list; every training sample contains all detectors, so max + 1 is its size).
        if "position_category_sizes" not in tokenizer_kwargs or (
            "position_continuous_dim" not in tokenizer_kwargs
        ):
            position = data_sample[2]  # [num_tokens, position_dim]
            position_dim = int(position.shape[-1])
            if "position_category_sizes" not in tokenizer_kwargs:
                position_continuous_dim = tokenizer_kwargs.get(
                    "position_continuous_dim", position_dim - 1
                )
                tokenizer_kwargs["position_category_sizes"] = [
                    int(position[:, c].max()) + 1
                    for c in range(position_continuous_dim, position_dim)
                ]
            tokenizer_kwargs.setdefault(
                "position_continuous_dim",
                position_dim - len(tokenizer_kwargs["position_category_sizes"]),
            )
        embedding_kwargs = model_kwargs["embedding_kwargs"]
        if embedding_kwargs.get("final_net_kwargs"):
            embedding_dim = embedding_kwargs["final_net_kwargs"]["output_dim"]
        else:
            embedding_dim = embedding_kwargs["transformer_kwargs"]["d_model"]
    else:
        model_kwargs["embedding_kwargs"]["input_dims"] = list(data_sample[1].shape)
        embedding_dim = model_kwargs["embedding_kwargs"]["output_dim"]
    model_kwargs["posterior_kwargs"]["context_dim"] = (
        embedding_dim + num_context_parameters
    )
