from copy import deepcopy

from hydra.utils import instantiate
from omegaconf import OmegaConf


def instantiate_with_runtime_dependencies(config, runtime_dependencies: dict):
    """
    Instantiate a Hydra config, optionally completing a partial with runtime objects.

    If the config contains ``_runtime_dependencies_``, only those named dependencies
    are passed to the partial. Otherwise all runtime dependencies are passed.
    """
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    else:
        config = deepcopy(config)
    dependency_names = config.pop("_runtime_dependencies_", None)
    obj = instantiate(config)
    if not config.get("_partial_", False):
        return obj

    if dependency_names is None:
        dependencies = runtime_dependencies
    else:
        dependencies = {
            name: runtime_dependencies[name]
            for name in dependency_names
        }
    return obj(**dependencies)
