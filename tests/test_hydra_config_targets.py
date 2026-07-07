from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from bilby.gw.prior import BBHPriorDict
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dingo.core.utils import get_optimizer_from_kwargs, get_scheduler_from_kwargs
from dingo.gw.SVD import SVDBasis
from dingo.gw.dataset import WaveformDataset
from dingo.gw.domains import (
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
    build_domain,
)
from dingo.gw.training.train_builders import set_train_transforms
from dingo.gw.training.train_pipeline import _populate_parameter_standardization
from dingo.gw.waveform_generator import WaveformGenerator


def compose_dingo_config(config_name: str, overrides: list[str] | None = None):
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize(
    ("domain_group", "expected_type"),
    [
        ("uniform_frequency", UniformFrequencyDomain),
        ("multibanded_frequency", MultibandedFrequencyDomain),
    ],
)
def test_domain_config_targets_instantiate(domain_group, expected_type):
    cfg = compose_dingo_config("generate_dataset", overrides=[f"domain={domain_group}"])
    domain_settings = OmegaConf.to_container(cfg.domain, resolve=True)

    assert isinstance(build_domain(domain_settings), expected_type)


def test_waveform_generator_config_target_instantiates_with_domain():
    cfg = compose_dingo_config("generate_dataset")
    domain = instantiate(cfg.domain)

    waveform_generator = instantiate(cfg.waveform_generator, domain=domain)

    assert isinstance(waveform_generator, WaveformGenerator)
    assert waveform_generator.domain is domain


@pytest.mark.parametrize(
    "prior_group",
    ["default", "precessing", "precessing_multibanded_test", "fmpe"],
)
def test_intrinsic_prior_config_targets_instantiate(prior_group):
    cfg = compose_dingo_config(
        "generate_dataset", overrides=[f"intrinsic_prior={prior_group}"]
    )

    prior = instantiate(cfg.intrinsic_prior)

    assert isinstance(prior, BBHPriorDict)
    assert "chirp_mass" in prior
    assert "mass_ratio" in prior


@pytest.mark.parametrize("prior_group", ["default", "fmpe"])
def test_extrinsic_prior_configs_build_prior_dict(prior_group):
    cfg = compose_dingo_config("train", overrides=[f"extrinsic_prior={prior_group}"])

    prior = instantiate(cfg.data.extrinsic_prior)

    assert set(prior) == {
        "dec",
        "ra",
        "geocent_time",
        "psi",
        "luminosity_distance",
    }


@pytest.mark.parametrize(
    "model_group",
    ["toy_npe", "npe", "gnpe", "gnpe_init", "fmpe", "unconditional_npe"],
)
def test_model_configs_define_posterior_model_target(model_group):
    cfg = compose_dingo_config("train", overrides=[f"model={model_group}"])

    assert cfg.model._target_.startswith("dingo.core.posterior_models.")
    assert "posterior_model_type" not in cfg.model


@pytest.mark.parametrize(
    "experiment",
    ["train_toy", "train_npe", "train_gnpe", "train_gnpe_init", "train_fmpe"],
)
def test_training_experiment_stage_optimizers_are_hydra_targets(experiment):
    cfg = compose_dingo_config("train", overrides=[f"+experiment={experiment}"])

    for stage_name, stage in cfg.training.items():
        if not stage_name.startswith("stage_"):
            continue
        assert "_target_" in stage.optimizer
        assert "_partial_" in stage.optimizer
        assert "_target_" in stage.scheduler
        assert "_partial_" in stage.scheduler


@pytest.mark.parametrize(
    ("experiment", "expected_targets"),
    [
        (
            None,
            [
                "dingo.gw.transforms.SampleExtrinsicParameters",
                "dingo.gw.transforms.GetDetectorTimes",
            ],
        ),
        (
            "train_gnpe",
            [
                "dingo.gw.transforms.SampleExtrinsicParameters",
                "dingo.gw.transforms.GetDetectorTimes",
                "dingo.gw.transforms.GNPECoalescenceTimes",
            ],
        ),
    ],
)
def test_training_standardization_transforms_are_configured(experiment, expected_targets):
    overrides = [] if experiment is None else [f"+experiment={experiment}"]
    cfg = compose_dingo_config("train", overrides=overrides)

    assert [t._target_ for t in cfg.data.standardization_transforms] == expected_targets

    if experiment == "train_gnpe":
        gnpe_transform = instantiate(cfg.data.standardization_transforms[-1])
        assert list(cfg.data.context_parameters) == gnpe_transform.context_parameters


def test_optimizer_scheduler_config_targets_instantiate():
    cfg = compose_dingo_config("train")
    network = torch.nn.Linear(2, 1)
    optimizer_settings = OmegaConf.to_container(
        cfg.training.stage_0.optimizer, resolve=True
    )
    scheduler_settings = OmegaConf.to_container(
        cfg.training.stage_0.scheduler, resolve=True
    )

    optimizer = get_optimizer_from_kwargs(
        network.parameters(),
        **optimizer_settings,
    )
    scheduler = get_scheduler_from_kwargs(optimizer, **scheduler_settings)

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_svd_basis_load_target_instantiates(tmp_path):
    file_name = tmp_path / "svd.hdf5"
    basis = SVDBasis()
    basis.generate_basis(np.eye(4), 2)
    basis.to_file(str(file_name))

    loaded = instantiate(
        {
            "_target_": "dingo.gw.dataset.compression.load_svd_basis",
            "file_name": str(file_name),
        }
    )

    assert isinstance(loaded, SVDBasis)
    assert loaded.V.shape == (4, 2)


def test_training_standardization_is_computed_before_transform_instantiation():
    cfg = compose_dingo_config(
        "train",
        overrides=[
            "data.inference_parameters=[chirp_mass,mass_ratio]",
            "model.embedding_kwargs.svd.num_training_samples=0",
        ],
    )
    settings = OmegaConf.to_container(cfg, resolve=True)

    domain_config = {
        "_target_": "dingo.gw.domains.UniformFrequencyDomain",
        "f_min": 0.0,
        "f_max": 4.0,
        "delta_f": 1.0,
    }
    domain = instantiate(domain_config)
    wfd = WaveformDataset(
        dictionary={
            "settings": {"domain": domain_config, "compression": None},
            "parameters": pd.DataFrame(
                {
                    "chirp_mass": [30.0, 31.0, 32.0],
                    "mass_ratio": [0.5, 0.6, 0.7],
                }
            ),
            "polarizations": {
                "h_plus": np.zeros((3, len(domain)), dtype=np.complex64),
                "h_cross": np.zeros((3, len(domain)), dtype=np.complex64),
            },
        }
    )
    settings["training"]["stage_0"]["asd_dataset"] = {
        "_target_": "dingo.gw.noise.asd_dataset.ASDDataset",
        "dictionary": {
            "settings": {"domain_dict": domain.domain_dict},
            "asds": {
                "H1": np.ones((2, len(domain))),
                "L1": np.ones((2, len(domain))),
            },
            "gps_times": {
                "H1": np.arange(2),
                "L1": np.arange(2),
            },
        },
        "ifos": ["H1", "L1"],
        "precision": "single",
        "domain_update": None,
    }

    _populate_parameter_standardization(
        wfd,
        settings["data"],
        settings["training"]["stage_0"]["asd_dataset"],
    )
    set_train_transforms(
        wfd,
        settings["data"],
        settings["training"]["stage_0"]["asd_dataset"],
    )

    assert settings["data"]["standardization"]["mean"] == {
        "chirp_mass": 31.0,
        "mass_ratio": 0.6,
    }
    assert list(settings["data"]["standardization"]["std"]) == [
        "chirp_mass",
        "mass_ratio",
    ]
    assert [type(t).__name__ for t in wfd.transform.transforms] == [
        "SampleExtrinsicParameters",
        "GetDetectorTimes",
        "ProjectOntoDetectors",
        "SampleNoiseASD",
        "WhitenAndScaleStrain",
        "AddWhiteNoiseComplex",
        "SelectStandardizeRepackageParameters",
        "RepackageStrainsAndASDS",
        "UnpackDict",
    ]
