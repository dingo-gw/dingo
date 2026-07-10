from dingo.gw.domains import (
    Domain,
    UniformFrequencyDomain,
    MultibandedFrequencyDomain,
    TimeDomain,
)
from hydra.utils import instantiate


def build_domain(settings: dict) -> Domain:
    """
    Instantiate a domain class from settings.

    Parameters
    ----------
    settings : dict
        Dictionary with 'type' key denoting the type of domain, and keys corresponding
        to the kwargs needed to construct the Domain.

    Returns
    -------
    A Domain instance of the correct type.
    """
    if "_target_" in settings:
        return instantiate(settings)

    if "type" not in settings:
        raise ValueError(
            'Domain settings must include a "_target_" or "type" key. '
            f"Settings included the keys {settings.keys()}."
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
    data_settings = model_metadata["train_settings"]["data"]
    domain_update = data_settings.get("waveform_dataset", {}).get(
        "domain_update",
        data_settings.get("domain_update"),
    )
    if domain_update is not None:
        domain.update(domain_update)
    if base and hasattr(domain, "base_domain"):
        domain = domain.base_domain
    return domain
