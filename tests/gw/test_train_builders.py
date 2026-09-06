"""Builder-to-loss check of the transformer training path.

Everything the training pipeline does before the first gradient step, on a fabricated
dataset: set_train_transforms with a full tokenization block, model autocompletion
from a sample, model construction, and one loss/backward pass. The detector list is
deliberately not in H1, L1, V1 order so that list-position indexing is exercised.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.training.train_builders import set_train_transforms

DETECTORS = ["V1", "H1"]
DOMAIN = {
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 64.0,
    "delta_f": 0.25,
}
TOKEN_SIZE = 16  # 177 bins in band -> 12 tokens per detector, last one zero-padded

DATA_SETTINGS = {
    "detectors": DETECTORS,
    "extrinsic_prior": {
        "dec": -1.21,
        "ra": 1.68,
        "geocent_time": 0.0,
        "psi": 0.0,
        "luminosity_distance": 439.0,
    },
    "ref_time": 1126259462.391,
    "inference_parameters": ["chirp_mass", "mass_ratio"],
    "tokenization": {
        "token_size": TOKEN_SIZE,
        "mask_detectors": {
            "p_num_masked": [0.5, 0.5],
            "p_detector": {"V1": 0.7, "H1": 0.3},
        },
        "mask_frequency_range": {
            "p_mask": 0.3,
            "f_min_upper": 30.0,
            "f_max_lower": 50.0,
            "p_lower_upper_both": [0.4, 0.4, 0.2],
            "p_same_all_detectors": 0.7,
        },
        "mask_frequency_notches": {"p_per_detector": 0.3, "max_width": 5.0},
        "mask_random_tokens": {"p_mask": 0.2, "max_num_tokens": 4},
    },
}

MODEL_SETTINGS = {
    "posterior_model_type": "normalizing_flow",
    "posterior_kwargs": {
        "num_flow_steps": 2,
        "base_transform_kwargs": {
            "hidden_dim": 16,
            "num_transform_blocks": 1,
            "activation": "elu",
            "batch_norm": False,
            "layer_norm": True,
            "dropout_probability": 0.0,
            "num_bins": 4,
            "base_transform_type": "rq-coupling",
        },
    },
    "embedding_type": "transformer",
    "embedding_kwargs": {
        "tokenizer_kwargs": {
            "hidden_dims": [32],
            "activation": "elu",
            "layer_norm": True,
        },
        "transformer_kwargs": {
            "d_model": 32,
            "dim_feedforward": 64,
            "nhead": 4,
            "dropout": 0.0,
            "num_layers": 1,
            "norm_first": True,
        },
        "pooling": "cls",
        "final_net_kwargs": {"output_dim": 16},
    },
}


def _toy_waveform_dataset(num_samples=8):
    """In-memory dataset with random polarizations on the full grid [0, f_max]."""
    rng = np.random.default_rng(0)
    num_bins = int(DOMAIN["f_max"] / DOMAIN["delta_f"]) + 1
    polarizations = {
        k: 1e-22
        * (
            rng.standard_normal((num_samples, num_bins))
            + 1j * rng.standard_normal((num_samples, num_bins))
        )
        for k in ("h_plus", "h_cross")
    }
    parameters = pd.DataFrame(
        {
            "chirp_mass": rng.uniform(20.0, 40.0, num_samples),
            "mass_ratio": rng.uniform(0.5, 1.0, num_samples),
            "luminosity_distance": 439.0,
            "geocent_time": 0.0,
        }
    )
    return WaveformDataset(
        dictionary={
            "settings": {"domain": DOMAIN},
            "parameters": parameters,
            "polarizations": polarizations,
        },
        precision="single",
    )


def _toy_asd_file(path):
    """ASD dataset for DETECTORS (plus an unused L1) on a grid covering DOMAIN."""
    freqs = np.arange(0.0, DOMAIN["f_max"] + DOMAIN["delta_f"] / 2, DOMAIN["delta_f"])
    with h5py.File(path, "w") as f:
        asds, gps = f.create_group("asds"), f.create_group("gps_times")
        for ifo, n in {"V1": 3, "H1": 4, "L1": 2}.items():
            asds.create_dataset(ifo, data=np.full((n, len(freqs)), 1e-23))
            gps[ifo] = np.arange(n)
        f.attrs["settings"] = str({"domain_dict": DOMAIN})
    return str(path)


def test_transformer_training_path_builder_to_loss(tmp_path):
    np.random.seed(0)
    torch.manual_seed(0)
    wfd = _toy_waveform_dataset()
    train_settings = {
        "data": {"waveform_dataset_path": None, **DATA_SETTINGS},
        "model": MODEL_SETTINGS,
        "training": {
            "stage_0": {"asd_dataset_path": _toy_asd_file(tmp_path / "asds.hdf5")}
        },
    }
    set_train_transforms(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
    )

    theta, waveform, position, token_mask = wfd[0]
    num_tokens_per_detector = wfd.domain.frequency_mask_length // TOKEN_SIZE + 1
    assert waveform.shape == (len(DETECTORS) * num_tokens_per_detector, 3 * TOKEN_SIZE)
    assert position.shape == (len(DETECTORS) * num_tokens_per_detector, 3)
    assert token_mask.shape == (len(DETECTORS) * num_tokens_per_detector,)
    assert token_mask.dtype == bool
    # Detector index = position in the training list, blocks in list order.
    expected = np.repeat(np.arange(len(DETECTORS)), num_tokens_per_detector)
    assert np.array_equal(position[:, 2], expected)

    autocomplete_model_kwargs(train_settings["model"], wfd[0])
    tokenizer_kwargs = train_settings["model"]["embedding_kwargs"]["tokenizer_kwargs"]
    assert tokenizer_kwargs["num_blocks"] == len(DETECTORS)
    assert tokenizer_kwargs["input_dim"] == 3 * TOKEN_SIZE

    pm = build_model_from_kwargs(
        settings={"dataset_settings": wfd.settings, "train_settings": train_settings},
        device="cpu",
    )
    data = next(iter(torch.utils.data.DataLoader(wfd, batch_size=4)))
    loss = pm.loss(data[0], *data[1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in pm.network.parameters()
    )
