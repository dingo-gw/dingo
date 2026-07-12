import numpy as np
import pandas as pd
import pytest

from dingo.core.density.unconditional_density_estimation import (
    train_unconditional_density_estimator,
)
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.result import Result


PARAMETERS = ["x", "y"]


def _nde_settings():
    """Minimal but valid settings for a fast (1-epoch) unconditional flow.

    Deliberately uses the old ``model.posterior_kwargs`` schema, so that the
    update_model_config shim inside train_unconditional_density_estimator is
    exercised (user nde settings files in the wild still use it).
    """
    return {
        "data": {},
        "model": {
            "posterior_model_type": "normalizing_flow",
            "posterior_kwargs": {
                "num_flow_steps": 2,
                "base_transform_kwargs": {
                    "hidden_dim": 8,
                    "num_transform_blocks": 1,
                    "activation": "elu",
                    "dropout_probability": 0.0,
                    "batch_norm": False,
                    "num_bins": 4,
                    "base_transform_type": "rq-coupling",
                },
            },
        },
        "training": {
            "device": "cpu",
            "num_workers": 0,
            "train_fraction": 0.9,
            "batch_size": 64,
            "epochs": 1,
            "optimizer": {"type": "adam", "lr": 0.01},
            "scheduler": {"type": "cosine", "T_max": 1},
        },
    }


@pytest.fixture()
def result():
    """A Result holding Gaussian samples over three parameters."""
    samples = pd.DataFrame(
        {
            "x": np.random.normal(1.0, 2.0, 256),
            "y": np.random.normal(-3.0, 0.5, 256),
            "z": np.random.normal(0.0, 1.0, 256),
        }
    )
    return Result(
        dictionary={"samples": samples, "settings": {"train_settings": {"data": {}}}}
    )


def test_train_unconditional_density_estimator_basic(result, tmp_path):
    settings = _nde_settings()
    settings["data"]["parameters"] = PARAMETERS
    model = train_unconditional_density_estimator(result, settings, str(tmp_path))

    assert isinstance(model, NormalizingFlowPosteriorModel)

    # The network is configured as an unconditional flow over the chosen parameters.
    assert settings["data"]["unconditional"] is True
    assert settings["model"]["distribution"]["kwargs"]["theta_dim"] == len(PARAMETERS)
    assert settings["model"]["distribution"]["kwargs"]["context_dim"] is None

    # Standardization is computed from the training samples.
    expected_mean = result.samples[PARAMETERS].to_numpy().mean(axis=0)
    expected_std = result.samples[PARAMETERS].to_numpy().std(axis=0)
    stored = settings["data"]["standardization"]
    for i, p in enumerate(PARAMETERS):
        assert stored["mean"][p] == pytest.approx(expected_mean[i])
        assert stored["std"][p] == pytest.approx(expected_std[i])

    # The trained model can sample with the right shapes.
    theta, log_prob = model.sample_and_log_prob(num_samples=5)
    assert tuple(theta.shape) == (5, len(PARAMETERS))
    assert tuple(log_prob.shape) == (5,)


def test_train_unconditional_density_estimator_uses_all_parameters_by_default(
    result, tmp_path
):
    # With no "parameters" entry, all sample columns are used.
    settings = _nde_settings()
    train_unconditional_density_estimator(result, settings, str(tmp_path))
    assert (
        settings["model"]["distribution"]["kwargs"]["theta_dim"]
        == result.samples.shape[1]
    )


def test_train_unconditional_flow_end_to_end(result, tmp_path):
    model = result.train_unconditional_flow(
        PARAMETERS, _nde_settings(), train_dir=str(tmp_path), threshold_std=np.inf
    )
    assert isinstance(model, NormalizingFlowPosteriorModel)
    # Trained over the requested subset only.
    assert model.model_kwargs["distribution"]["kwargs"]["theta_dim"] == len(PARAMETERS)


def test_train_unconditional_flow_rejects_too_many_outliers(result):
    # A tight threshold removes far more than 5% of Gaussian samples, so the outlier
    # guard raises before any training happens (no train_dir needed).
    with pytest.raises(ValueError, match="Too many proxy samples"):
        result.train_unconditional_flow(PARAMETERS, _nde_settings(), threshold_std=1.0)
