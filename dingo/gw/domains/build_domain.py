from dingo.gw.domains import (
    Domain,
    UniformFrequencyDomain,
    MultibandedFrequencyDomain,
    TimeDomain,
)
from dingo.gw.domains.base import DomainParameters
from dingo.gw.imports import import_entity

_module_import_path = "dingo.gw.domains"


def build_domain(settings) -> Domain:
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    settings : dict or DomainParameters
        Dictionary with 'type' key denoting the type of domain, and keys corresponding
        to the kwargs needed to construct the Domain, OR a DomainParameters instance.

    Returns
    -------
    A Domain instance of the correct type.
    """
    if isinstance(settings, DomainParameters):
        return _build_domain_from_parameters(settings)
    elif isinstance(settings, dict):
        return _build_domain_from_dict(settings)
    else:
        raise TypeError(
            f"Expected dict or DomainParameters, got {type(settings)}."
        )


def _build_domain_from_dict(settings: dict) -> Domain:
    """Build domain from a settings dictionary."""
    if "type" not in settings:
        raise ValueError(
            f'Domain settings must include a "type" key. Settings included '
            f"the keys {settings.keys()}."
        )

    # The settings other than 'type' correspond to the kwargs of the Domain constructor.
    kwargs = {k: v for k, v in settings.items() if k != "type"}
    if settings["type"] in ["UniformFrequencyDomain", "FrequencyDomain", "FD"]:
        return UniformFrequencyDomain(**kwargs)
    elif settings["type"] in ["MultibandedFrequencyDomain", "MFD"]:
        return MultibandedFrequencyDomain(**kwargs)
    elif settings["type"] in ["TimeDomain", "TD"]:
        return TimeDomain(**kwargs)
    else:
        raise NotImplementedError(f'Domain {settings["type"]} not implemented.')


def _build_domain_from_parameters(domain_parameters: DomainParameters) -> Domain:
    """Build domain from a DomainParameters instance."""
    if domain_parameters.type is None:
        raise ValueError(
            "Constructing domain: 'type' should not be None in DomainParameters."
        )

    # If type is a short name, resolve it to a full import path
    type_str = domain_parameters.type
    if "." not in type_str:
        type_str = f"{_module_import_path}.{type_str}"

    class_, _, class_name = import_entity(type_str)

    if not issubclass(class_, Domain):
        raise ValueError(
            f"Constructing domain: could import '{type_str}', "
            "but this is not a subclass of 'Domain'"
        )

    try:
        instance = class_.from_parameters(domain_parameters)
    except Exception as e:
        raise RuntimeError(
            f"Constructing domain: failed to construct {class_name} "
            f"from DomainParameters. {type(e)}: {e}"
        ) from e

    return instance


def build_domain_from_model_metadata(
    model_metadata: dict, base: bool = False
) -> Domain:
    """
    Instantiate a domain class from settings of model.

    Parameters
    ----------
    model_metadata: dict
        model metadata containing information to build the domain
        typically obtained from the model.metadata attribute
    base: bool = False
        If base=True, return domain.base_domain if this is an attribute of domain,
        else return domain. Example: MultibandedFrequencyDomain has a UniformFrequencyDomain
        object as a base_domain. In dingo_pipe, we want to load data in the
        base_domain, and later decimate from base_domain to domain.

    Returns
    -------
    A Domain instance of the correct type.
    """
    domain = build_domain(model_metadata["dataset_settings"]["domain"])
    if "domain_update" in model_metadata["train_settings"]["data"]:
        domain.update(model_metadata["train_settings"]["data"]["domain_update"])
    if base and hasattr(domain, "base_domain") and domain.base_domain is not None:
        domain = domain.base_domain
    return domain
