# Importing the modules registers the built-in model types with NEURAL_DISTRIBUTIONS.
import dingo.core.posterior_models.flow_matching  # noqa: F401
import dingo.core.posterior_models.normalizing_flow  # noqa: F401
import dingo.core.posterior_models.score_matching  # noqa: F401
from dingo.core.posterior_models.base_model import NeuralDistribution
from dingo.core.registry import NEURAL_DISTRIBUTIONS
from dingo.core.utils.backward_compatibility import (
    torch_load_with_fallback,
    update_model_config,
    check_minimum_version,
)


def build_model_from_kwargs(
    filename: str = None, settings: dict = None, **kwargs
) -> NeuralDistribution:
    """
    Returns a NeuralDistribution based on a saved network or settings dict.

    The model class is resolved from the settings' posterior_model_type via the
    NEURAL_DISTRIBUTIONS registry (e.g., normalizing flow, flow matching, or score
    matching, or a plugin type; see dingo.core.registry).

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
    NeuralDistribution
    """
    if (filename is None) == (settings is None):
        raise ValueError(
            "Either a filename or a settings dict must be provided, but not both."
        )

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
        posterior_model_type = settings["train_settings"]["model"][
            "posterior_model_type"
        ]

    try:
        model = NEURAL_DISTRIBUTIONS.get(posterior_model_type)
    except KeyError as e:
        raise ValueError(f"No valid posterior model type specified. {e}") from e

    return model(model_filename=filename, metadata=settings, **kwargs)


def autocomplete_model_kwargs(model_kwargs: dict, data_sample: dict):
    """
    Autocomplete the model kwargs from train_settings and data_sample from the dataloader:

    * set input dimension of embedding net to the shape of the waveform data
    * set dimension of parameter space to the number of inference parameters
    * set added_context flag of embedding net if required for context parameters
      (e.g., GNPE proxies)
    * set context dim of posterior model to output dim of embedding net + dimension
      of the context parameters

    Parameters
    ----------
    model_kwargs: dict
        Model settings, which are modified in-place.
    data_sample: dict
        Sample from dataloader (e.g., wfd[0]) used for autocompletion, with keys
        "inference_parameters", "waveform", and (only if the network is conditioned
        on additional parameters) "context_parameters".
    """

    # set input dims from ifo_list and domain information
    model_kwargs["embedding_kwargs"]["input_dims"] = list(data_sample["waveform"].shape)
    # set dimension of parameter space of posterior model
    model_kwargs["posterior_kwargs"]["input_dim"] = len(
        data_sample["inference_parameters"]
    )
    # set added_context flag of embedding net if context parameters are required
    # set context dim of nsf to output dim of embedding net + context parameter dim
    if "context_parameters" in data_sample:
        model_kwargs["embedding_kwargs"]["added_context"] = True
        model_kwargs["posterior_kwargs"]["context_dim"] = model_kwargs[
            "embedding_kwargs"
        ]["output_dim"] + len(data_sample["context_parameters"])
    else:
        model_kwargs["embedding_kwargs"]["added_context"] = False
        model_kwargs["posterior_kwargs"]["context_dim"] = model_kwargs[
            "embedding_kwargs"
        ]["output_dim"]
