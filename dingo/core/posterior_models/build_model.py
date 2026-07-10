from dingo.core.posterior_models.base_model import BasePosteriorModel
from hydra.utils import instantiate
from dingo.core.utils.backward_compatibility import (
    torch_load_with_fallback,
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

    if filename is not None:
        d, _ = torch_load_with_fallback(filename, preferred_map_location="meta")
        if "version" in d:
            check_minimum_version(d["version"])
        else:
            # version was introduced in v0.3.3
            check_minimum_version("dingo=0.3.2")
        update_model_config(d["metadata"]["train_settings"]["model"])  # Backward compat
        model_settings = d["metadata"]["train_settings"]["model"]
    else:
        update_model_config(settings["train_settings"]["model"])  # Backward compat
        model_settings = settings["train_settings"]["model"]

    if "_target_" not in model_settings:
        raise KeyError("Model settings need a Hydra _target_.")

    model_target = {"_target_": model_settings["_target_"]}
    return instantiate(model_target, model_filename=filename, metadata=settings, **kwargs)
