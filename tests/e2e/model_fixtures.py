"""Metadata-only checkpoints and a synthetic PSD for the Asimov smoke test."""

import copy
from pathlib import Path

import numpy as np
import torch

from dingo.gw.domains import build_domain


def create_model_checkpoints(
    event: dict, settings: dict, directory: Path
) -> dict[str, str]:
    """Save only metadata; no network is constructed or serialized."""
    checkpoints = {}
    for name in ("transformer", "init", "gnpe"):
        metadata = build_checkpoint_metadata(name, event, settings)
        path = str(directory / f"{name}.pt")
        torch.save({"metadata": metadata}, path)
        checkpoints[name] = path
    return checkpoints


def write_noise_psd(settings: dict, path: Path) -> str:
    """Write a flat synthetic PSD for configuration and file-transfer coverage."""
    domain = build_domain(settings["dataset_settings"]["domain"]).base_domain
    np.savetxt(path, np.column_stack((domain(), np.full(len(domain), 1e-46))))
    return str(path)


def build_checkpoint_metadata(name: str, event: dict, settings: dict) -> dict:
    metadata = {
        key: copy.deepcopy(settings[key])
        for key in ("dataset_settings", "train_settings")
    }
    data = metadata["train_settings"]["data"]
    data.update(detectors=event["interferometers"], ref_time=event["event time"])
    configure_preprocessing(data, name)

    metadata["train_settings"]["model"].update(
        embedding_type="transformer" if name == "transformer" else "resnet",
        embedding_kwargs={},
    )
    return metadata


def configure_preprocessing(data: dict, name: str) -> None:
    """Describe the model's preprocessing; no training data is generated."""
    data["context_parameters"] = []
    if name == "transformer":
        data["tokenization"] = {
            "token_size": 16,
            "normalize_position": True,
            "mask_frequency_range": {
                "p_mask": 0.5,
                "f_min_upper": 32.0,
                "f_max_lower": 80.0,
                "p_lower_upper_both": [0.4, 0.4, 0.2],
                "p_same_all_detectors": 0.5,
            },
        }
        return

    data["random_strain_cropping"] = {
        "cropping_probability": 0.5,
        "f_min_upper": 32.0,
        "f_max_lower": 80.0,
        "independent_detectors": True,
    }
    if name == "init":
        # The first GNPE kernel step conditions on detector arrival times.
        parameters = [f"{detector}_time" for detector in data["detectors"]]
        data["inference_parameters"] = parameters
        data["standardization"] = {
            "mean": {parameter: 0.0 for parameter in parameters},
            "std": {parameter: 0.01 for parameter in parameters},
        }
    elif name == "gnpe":
        data["gnpe_time_shifts"] = {
            "kernel": "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)",
            "exact_equiv": True,
        }
        data["context_parameters"] = ["L1_time_proxy_relative"]
        data["standardization"]["mean"]["L1_time_proxy_relative"] = 0.0
        data["standardization"]["std"]["L1_time_proxy_relative"] = 0.01
        data["standardization"]["std"]["geocent_time"] = 0.003
