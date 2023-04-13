from .base import Domain
from .frequency_domain import FrequencyDomain
from .time_domain import TimeDomain
from .multibanded_frequency_domain import (
    MultibandedFrequencyDomain,
    get_decimation_bands_adaptive,
)

from copy import deepcopy
import numpy as np


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
    if "type" not in settings:
        raise ValueError(
            f'Domain settings must include a "type" key. Settings included '
            f"the keys {settings.keys()}."
        )

    # The settings other than 'type' correspond to the kwargs of the Domain constructor.
    kwargs = {k: v for k, v in settings.items() if k != "type"}
    if settings["type"] in ["FrequencyDomain", "FD"]:
        return FrequencyDomain(**kwargs)
    elif settings["type"] in ["MultibandedFrequencyDomain", "MFD"]:
        return MultibandedFrequencyDomain(**kwargs)
    elif settings["type"] == ["TimeDomain", "TD"]:
        return TimeDomain(**kwargs)
    else:
        raise NotImplementedError(f'Domain {settings["name"]} not implemented.')


def build_domain_from_wfd_settings(settings: dict, num_processes: int = 0) -> Domain:
    """
    Instantiate a domain class from waveform dataset settings.

    - In most cases, this just calls build_domain(settings["domain"]).
    - If domain type is MultiBandedFrequency domain and the bands argument is not
      provided, this function generates a small waveform dataset to determine the bands.
      This also changes the domain settings *in place*, such that generated bands are
      loaded in the future, and not regenerated.

    Parameters
    ----------
    settings : dict
        Dictionary with waveform dataset settings.
        In particular, settings["domain"] contains the domain settings which can be
        used to call build_domain(settings["domain"]).
        Moreover, a full waveform dataset can be generated with
        dingo.gw.dataset.generate_dataset(settings, num_processes).
    num_processes : int = 0
        Number of processes for waveform generation (if required).

    Returns
    -------
    A Domain instance of the correct type.
    """
    if (
        settings["domain"]["type"] == "MultibandedFrequencyDomain"
        and "bands" not in settings["domain"]
    ):
        settings_mfd = deepcopy(settings["domain"])
        settings_mfd.pop("type")
        multibanding_method = settings_mfd.pop("multibanding_method", "adaptive")

        if multibanding_method == "adaptive":
            from dingo.gw.dataset import generate_dataset

            # Generate polarizations in uniform FrequencyDomain
            print(
                "Generating polarizations to determine suitable bands for the "
                "MultibandedFrequencyDomain."
            )
            settings_ufd = deepcopy(settings)
            settings_ufd["num_samples"] = settings_mfd.pop(
                "num_samples_band_generation"
            )
            settings_ufd["domain"] = settings_mfd.pop("base_domain")
            # We don't need compression to generate just a few waveforms for this step.
            if "compression" in settings_ufd:
                settings_ufd["compression"].pop("svd", None)
            wfd = generate_dataset(settings_ufd, num_processes)
            # We now extract the polarizations directly, and not via wfd[idx] (similarly
            # to what's done in train_svd_basis in dingo.gw.dataset.generate_datset).
            # This means that decompression transforms (in particular phase_heterodyning)
            # will *not* be applied, which is desirable if the final waveform passed to
            # the network will be transformed in a similar way (e.g., GNPE for phase
            # heterodyning).
            polarizations = np.concatenate(list(wfd.polarizations.values()))

            nodes_f, initial_delta_f = get_decimation_bands_adaptive(
                wfd.domain, polarizations, **settings_mfd
            )
        else:
            raise NotImplementedError(
                f"Unknown multibanding method {multibanding_method}."
            )
        mfd = MultibandedFrequencyDomain(nodes_f, initial_delta_f, wfd.domain)
        print(f"MultibandedFrequencyDomain:\n{mfd.domain_dict}")
        return mfd

    else:
        return build_domain(settings["domain"])


def build_domain_from_model_metadata(model_metadata) -> Domain:
    """
    Instantiate a domain class from settings of model.

    Parameters
    ----------
    model_metadata: dict
        model metadata containing information to build the domain
        typically obtained from the model.metadata attribute

    Returns
    -------
    A Domain instance of the correct type.
    """
    domain = build_domain(model_metadata["dataset_settings"]["domain"])
    if "domain_update" in model_metadata["train_settings"]["data"]:
        domain.update(model_metadata["train_settings"]["data"]["domain_update"])
    domain.window_factor = get_window_factor(
        model_metadata["train_settings"]["data"]["window"]
    )
    return domain
