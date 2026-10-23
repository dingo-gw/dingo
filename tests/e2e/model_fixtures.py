"""Tiny initialized checkpoints and an analytic PSD for the Asimov smoke test."""

import copy
from pathlib import Path

import numpy as np
import torch
from bilby.gw.detector import PowerSpectralDensity

from dingo.core.posterior_models import build_model_from_kwargs
from dingo.core.utils.torchutils import seed_generators
from dingo.gw.domains import build_domain


def create_model_checkpoints(
    event: dict, settings: dict, directory: Path
) -> dict[str, str]:
    """Save untrained transformer, GNPE initialization, and GNPE checkpoints."""
    torch.set_num_threads(1)
    seed_generators(np.random.SeedSequence(42))
    checkpoints = {}
    for name in ("transformer", "init", "gnpe"):
        metadata = build_checkpoint_metadata(name, event, settings)
        model = build_model_from_kwargs(settings=metadata, device="cpu")
        path = str(directory / f"{name}.pt")
        model.save_model(path, save_training_info=False)
        checkpoints[name] = path
    return checkpoints


def write_noise_psd(settings: dict, path: Path) -> str:
    """Write the analytic PSD on the uniform grid required by Bilby's generator."""
    domain = build_domain(settings["dataset_settings"]["domain"]).base_domain
    psd = PowerSpectralDensity(asd_file="aLIGO_ZERO_DET_high_P_asd.txt")
    asd = np.interp(domain(), psd.frequency_array, psd.asd_array, left=1, right=1)
    np.savetxt(path, np.column_stack((domain(), asd**2)))
    return str(path)


def build_checkpoint_metadata(name: str, event: dict, settings: dict) -> dict:
    metadata = {
        key: copy.deepcopy(settings[key])
        for key in ("dataset_settings", "train_settings")
    }
    data = metadata["train_settings"]["data"]
    data.update(detectors=event["interferometers"], ref_time=event["event time"])
    configure_preprocessing(data, name)

    embedding_type = "transformer" if name == "transformer" else "resnet"
    embedding = copy.deepcopy(settings[embedding_type])
    if embedding_type == "transformer":
        embedding["tokenizer_kwargs"].update(
            input_dim=3 * data["tokenization"]["token_size"],
            position_continuous_dim=2,
            position_category_sizes=[len(data["detectors"])],
        )
        embedding_dimension = embedding["final_net_kwargs"]["output_dim"]
    else:
        domain = build_domain(metadata["dataset_settings"]["domain"])
        embedding.update(
            input_dims=[len(data["detectors"]), 3, domain.frequency_mask_length],
            added_context=bool(data["context_parameters"]),
        )
        embedding_dimension = embedding["output_dim"]

    model = metadata["train_settings"]["model"]
    model.update(embedding_type=embedding_type, embedding_kwargs=embedding)
    model["posterior_kwargs"].update(
        input_dim=len(data["inference_parameters"]),
        context_dim=embedding_dimension + len(data["context_parameters"]),
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
