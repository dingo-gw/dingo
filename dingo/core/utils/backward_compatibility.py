import logging
from typing import Dict, List, Literal, Tuple
from dingo.core.utils.misc import get_version
import packaging.version as pv
import warnings

import torch

_logger = logging.getLogger(__name__)
WINDOW_FACTOR_FIX_VERSION = pv.parse("0.9.0")

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
    except (RuntimeError, AttributeError):
        # AttributeError can occur due to PyTorch bug: when CUDA is requested
        # on Mac, PyTorch's internal fallback tries torch.mps.current_device()
        # which doesn't exist (torch.mps lacks feature parity with torch.cuda)
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
        except (RuntimeError, AttributeError):
            pass

    raise RuntimeError(
        f"failed to load model {filename} on any device, " "tried: {', '.join(devices)}"
    )


def check_minimum_version(version_str: str, raise_exception: bool = False) -> None:
    """
    Check that the version string is greater than a certain minimum value.

    By default, logs a warning. Optionally, raises an exception.

    This is used to handle major code changes that may break backwards compatibility
    with previously trained models or generated results.

    Parameters
    ----------
    version_str : str
        Version string to check, e.g., "version=0.8.5" or "0.8.5".

    raise_exception : bool
        If True, raise an exception if the version is below the minimum required version.
    """
    if "None" in version_str:
        version_str = "dingo=0.0.0"
    version_str = version_str.split("=", 1)[1]
    version = pv.parse(version_str)

    if version < WINDOW_FACTOR_FIX_VERSION:
        error_str = (
            f"This object was created using Dingo version {version} < {WINDOW_FACTOR_FIX_VERSION}, which broke backwards compatibility."
            f"\nFor models trained prior to this change, new inference results will be unreliable."
            f"\nPreviously-generated result files should be used with caution."
            f"\nReasons for backward compatibility breaking:\n"
            f"\nv{WINDOW_FACTOR_FIX_VERSION}: Change to window factor usage, see "
            f"https://git.ligo.org/pe/pe-group-coordination/-/issues/1#note_1469386."
        )
        if raise_exception:
            raise ValueError(error_str)
        else:
            _logger.warning("\n========\nWARNING!\n\n" + error_str + "\n=======\n")


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

    if "embedding_kwargs" in model_settings and "embedding_type" not in model_settings:
        model_settings["embedding_type"] = "resnet"

    if model_settings.get("embedding_type") == "transformer":
        # Networks trained on the dingo-t1 branch (e.g. the published Dingo-T1
        # network) store kwargs that are now either fixed or derived.
        embedding_kwargs = model_settings["embedding_kwargs"]
        embedding_kwargs.pop("added_context", None)
        tokenizer_kwargs = embedding_kwargs.get("tokenizer_kwargs") or {}
        if not tokenizer_kwargs.pop(
            "condition_on_position", True
        ) or tokenizer_kwargs.pop("context_in_initial_layer", False):
            raise ValueError(
                "Transformer networks with condition_on_position=False or "
                "context_in_initial_layer=True (dingo-t1 branch) are not supported."
            )
        # Derived from num_blocks and d_model at construction.
        for key in ("context_features", "output_dim"):
            tokenizer_kwargs.pop(key, None)
        # [num_tokens, num_features] -> num_features (only the latter was used).
        if "input_dims" in tokenizer_kwargs:
            tokenizer_kwargs["input_dim"] = tokenizer_kwargs.pop("input_dims")[-1]
        (embedding_kwargs.get("final_net_kwargs") or {}).pop("input_dim", None)


def update_data_config(settings: dict):
    """
    Update ``settings["train_settings"]["data"]`` to the current keys, in place.
    Renames the tokenization settings written by the dingo-t1 branch (e.g. the
    published Dingo-T1 network), filling in the constant defaults that branch
    applied for absent keys. Idempotent.

    Parameters
    ----------
    settings: dict
        Model metadata or training settings, i.e. a dict with ``train_settings``.
    """
    data_settings = settings.get("train_settings", {}).get("data", {})
    tok = data_settings.get("tokenization")
    if tok is None:
        return
    if tok.pop("normalize_frequency_for_positional_encoding", False):
        raise NotImplementedError(
            "Networks trained with normalize_frequency_for_positional_encoding=True "
            "(dingo-t1 branch) are not supported."
        )
    if "num_tokens" in tok:
        tok["num_tokens_per_block"] = tok.pop("num_tokens")
    if "drop_detectors" in tok:
        old = tok.pop("drop_detectors")
        tok["mask_detectors"] = {
            "num_blocks": len(data_settings["detectors"]),
            "p_mask_012_detectors": old.get("p_drop_012_detectors"),
            "p_mask_hlv": old.get("p_drop_hlv"),
        }
    if "drop_frequency_range" in tok:
        old = tok.pop("drop_frequency_range")
        if "f_cut" in old:
            f_cut = old["f_cut"]
            frequency_range = {
                "p_mask": f_cut.get("p_cut", 0.2),
                "p_same_all_detectors": f_cut.get("p_same_cut_all_detectors", 0.2),
                "p_lower_upper_both": f_cut.get("p_lower_upper_both", [0.4, 0.4, 0.2]),
            }
            # The bound names are deliberately swapped: mask_frequency_range uses
            # random_strain_cropping's convention (f_min_upper = cap on f_min,
            # f_max_lower = floor on f_max), the mirror image of dingo-t1's cut
            # names. Absent bounds stay absent (that side was never cut).
            for new, old_key in (
                ("f_min_upper", "f_max_lower_cut"),
                ("f_max_lower", "f_min_upper_cut"),
            ):
                if old_key in f_cut:
                    frequency_range[new] = f_cut[old_key]
            tok["mask_frequency_range"] = frequency_range
        if "mask_interval" in old:
            tok["mask_frequency_notches"] = {
                "p_per_detector": 0.2,
                **old["mask_interval"],
            }
    if "drop_random_tokens" in tok:
        old = tok.pop("drop_random_tokens")
        if old.get("increase_p_until_epoch") is not None:
            warnings.warn(
                "drop_random_tokens.increase_p_until_epoch is no longer supported "
                "and is ignored."
            )
        tok["mask_random_tokens"] = {
            "p_mask": old.get("p_drop", 0.4),
            "max_num_tokens": old.get(
                "max_num_tokens", tok.get("num_tokens_per_block")
            ),
        }
