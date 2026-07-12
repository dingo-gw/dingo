import copy

# Importing the modules registers the built-in model types with NEURAL_DISTRIBUTIONS.
import dingo.core.posterior_models.flow_matching  # noqa: F401
import dingo.core.posterior_models.normalizing_flow  # noqa: F401
import dingo.core.posterior_models.score_matching  # noqa: F401
from dingo.core.posterior_models.base_model import NeuralDistribution
from dingo.core.registry import (
    CONTEXT_MERGERS,
    EMBEDDING_NETS,
    NEURAL_DISTRIBUTIONS,
)
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

    The model class is resolved from the settings' distribution type via the
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
        model_type = d["metadata"]["train_settings"]["model"]["distribution"]["type"]
    else:
        update_model_config(settings["train_settings"]["model"])  # Backward compat
        model_type = settings["train_settings"]["model"]["distribution"]["type"]

    try:
        model = NEURAL_DISTRIBUTIONS.get(model_type)
    except KeyError as e:
        raise ValueError(f"No valid posterior model type specified. {e}") from e

    return model(model_filename=filename, metadata=settings, **kwargs)


def complete_model_settings(model_settings: dict, sample_batch: dict) -> dict:
    """
    Complete the model settings based on a sample batch from the dataloader.

    Each embedding architecture infers its own input dimensions via its
    complete_settings classmethod; the cross-cutting dimensions (theta_dim,
    context_dim) are computed here. The completed settings are saved in the
    checkpoint, so loading a model never needs a data sample.

    If the batch provides context_parameters and the embedding network does not
    consume them natively (i.e., they are not among its input_keys), a context
    merger is added ("concat" unless specified otherwise).

    Dimensions are derived from the data; specifying them in the settings is an
    error.

    Parameters
    ----------
    model_settings: dict
        Model section of the train settings. Old schemas are mapped forward
        in-place; the completion itself does not modify the input.
    sample_batch: dict
        Sample from the dataloader (e.g., wfd[0]), with keys
        "inference_parameters", "waveform", optionally "context_parameters", and
        any architecture-specific entries.

    Returns
    -------
    dict
        Completed model settings.
    """
    update_model_config(model_settings)
    model_settings = copy.deepcopy(model_settings)

    distribution_kwargs = model_settings["distribution"].setdefault("kwargs", {})
    for key in ("theta_dim", "context_dim"):
        if key in distribution_kwargs:
            raise ValueError(
                f"'{key}' is derived from the data and must not be specified in "
                f"the model settings."
            )
    distribution_kwargs["theta_dim"] = len(sample_batch["inference_parameters"])

    embedding_settings = model_settings.get("embedding_net")
    if embedding_settings is None:
        if "context_merger" in model_settings:
            raise ValueError("A context_merger requires an embedding_net.")
        distribution_kwargs["context_dim"] = None
        return model_settings

    embedding_cls = EMBEDDING_NETS.get(embedding_settings["type"])
    missing = [k for k in embedding_cls.input_keys if k not in sample_batch]
    if missing:
        raise ValueError(
            f"Embedding net '{embedding_settings['type']}' consumes batch entries "
            f"{list(embedding_cls.input_keys)}, but the sample batch is missing "
            f"{missing} (batch keys: {sorted(sample_batch)})."
        )
    embedding_settings["kwargs"] = embedding_cls.complete_settings(
        embedding_settings.get("kwargs", {}), sample_batch
    )
    output_dim = embedding_settings["kwargs"]["output_dim"]

    native_context = "context_parameters" in embedding_cls.input_keys
    if "context_parameters" in sample_batch and not native_context:
        merger_settings = model_settings.setdefault(
            "context_merger", {"type": "concat"}
        )
        merger_cls = CONTEXT_MERGERS.get(merger_settings["type"])
        merger_settings["kwargs"] = {
            **merger_settings.get("kwargs", {}),
            "num_context_parameters": len(sample_batch["context_parameters"]),
        }
        distribution_kwargs["context_dim"] = merger_cls.merged_output_dim(
            output_dim, **merger_settings["kwargs"]
        )
    else:
        if "context_merger" in model_settings:
            raise ValueError(
                "A context_merger is specified, but the data provides no "
                "context_parameters (or the embedding net consumes them natively)."
            )
        distribution_kwargs["context_dim"] = output_dim
    return model_settings
