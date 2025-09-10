import logging
from typing import Dict, List, Literal, Tuple
from dingo._version import __version__
import packaging.version as pv

import torch

_logger = logging.getLogger(__name__)

Device = Literal["meta", "cuda", "mps", "hip", "cpu"]


def torch_available_devices() -> List[Device]:
    """
    Returns a list of all available PyTorch devices,
    ordered: cuda, mps, hip, cpu
    Note: 'meta' is not included from the returned list,
    even if supported.

    Returns
    -------
    List of available device identifiers
    """
    devices: List[Device] = []

    # cuda
    if torch.cuda.is_available():
        devices.append("cuda")

    # mps
    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except AttributeError:
        pass

    # hip
    try:
        if hasattr(torch, "hip") and torch.hip.is_available():
            devices.append("hip")
    except AttributeError:
        pass

    # cpu
    devices.append("cpu")

    return devices


def torch_load_with_fallback(
    filename: str, preferred_map_location: Device = "cuda"
) -> Tuple[Dict, torch.device]:
    """
    Loads a PyTorch file with fallback behavior:
    1. Tries preferred_map_location (default: cuda)
    2. Falls back to CUDA/MPS/HIP if available
    3. Finally falls back to CPU

    Returns
    -------
    Loaded model and torch device on which it has been loaded
    """

    try:
        r = (
            torch.load(filename, map_location=preferred_map_location),
            torch.device(preferred_map_location),
        )
        _logger.debug(f"loaded model {filename} to {preferred_map_location}")
        return r
    except RuntimeError:
        pass

    devices = torch_available_devices()

    for location in [d for d in devices if d != preferred_map_location]:
        try:
            r = torch.load(filename, map_location=location), torch.device(location)
            _logger.debug(
                f"loaded model {filename} to fallback device {location} "
                f"(preferred device was {preferred_map_location})"
            )
            return r
        except RuntimeError:
            pass

    raise RuntimeError(
        f"failed to load model {filename} on any device, " "tried: {', '.join(devices)}"
    )


def check_window_factor_fix(model_metadata: dict) -> bool:
    """
    Versions of DINGO before 0.8.6 were using an erroneous computation of the
    window factor. For the full discussion see:
    https://git.ligo.org/pe/pe-group-coordination/-/issues/1#note_1469386
    Therefore, we if the version of the network does not match the version of the
    package, we raise an exception telling the user to either upgrade or
    downgrade their version of DINGO such that the code and networks
    are consistent.

    Parameters
    ----------
    model_metadata : dict
        Metdata of the DINGO model
    """
    dingo_version = pv.parse(__version__)
    model_version_str = model_metadata["version"].split("=", 1)[1]
    model_version = pv.parse(model_version_str)
    window_factor_fix_version = pv.parse("0.8.6")

    class VersionMismatchError(Exception):
        pass

    if (
        dingo_version < window_factor_fix_version
        and model_version >= window_factor_fix_version
    ):
        raise VersionMismatchError(
            f"""
        Your DINGO version ({dingo_version}) is before the window factor fix,
        but the model version ({model_version}) is after the window factor fix.
        Please upgrade your DINGO version to {window_factor_fix_version} or later
        to use this network.
        """.strip()
        )
    elif (
        dingo_version >= window_factor_fix_version
        and model_version < window_factor_fix_version
    ):
        raise VersionMismatchError(
            f"""
        Your DINGO version ({dingo_version}) is after the window factor fix and model version ({model_version})
        is before the window factor fix. Please downgrade your dingo version to before {window_factor_fix_version} 
        to use this network.
        """.strip()
        )


def update_model_config(model_settings: dict):
    """
    Update the model settings to ensure backwards compatibility with networks
    trained using previous versions of Dingo.

    Parameters
    ----------
    model_settings: dict
        Model settings to be updated.
    """
    if model_settings.get("type") == "nsf+embedding":
        model_settings["posterior_model_type"] = "normalizing_flow"
        del model_settings["type"]
        model_settings["posterior_kwargs"] = model_settings["nsf_kwargs"]
        del model_settings["nsf_kwargs"]
        model_settings["embedding_kwargs"] = model_settings["embedding_net_kwargs"]
        del model_settings["embedding_net_kwargs"]
