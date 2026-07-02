"""
Characterization tests for the model build path.

These tests pin the *current* behavior of building posterior models from settings
dictionaries — type dispatch (build_model_from_kwargs), dimensional autocompletion
(autocomplete_model_kwargs), end-to-end construction and forward passes for all three
posterior model types, SVD initial-weight seeding, and loading of old-schema
checkpoints — ahead of the NN build-system refactor (see hackathon/
NN_Build_System_Design.md). If one of these tests breaks, either the refactor changed
observable behavior (fix the refactor) or the behavior change is intended and
documented (update the test alongside the compatibility shim).
"""

import copy

import numpy as np
import pytest
import torch

from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.core.posterior_models.flow_matching import FlowMatchingPosteriorModel
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.score_matching import ScoreDiffusionPosteriorModel
from dingo.core.utils.backward_compatibility import update_model_config

# Data dimensions used throughout: 4 inference parameters, strain data of shape
# (num_blocks=2, num_channels=3, num_bins=20), embedding output dimension 8.
NUM_PARAMETERS = 4
DATA_SHAPE = (2, 3, 20)
EMBEDDING_OUTPUT_DIM = 8
GNPE_PROXY_DIM = 2
BATCH_SIZE = 5


def embedding_kwargs():
    return {
        "input_dims": list(DATA_SHAPE),
        "svd": {"size": 10},
        "V_rb_list": None,
        "output_dim": EMBEDDING_OUTPUT_DIM,
        "hidden_dims": [32, 16, 8],
        "activation": "elu",
        "dropout": 0.0,
        "batch_norm": True,
        "added_context": False,
    }


def nsf_posterior_kwargs():
    return {
        "input_dim": NUM_PARAMETERS,
        "context_dim": EMBEDDING_OUTPUT_DIM,
        "num_flow_steps": 2,
        "base_transform_kwargs": {
            "hidden_dim": 16,
            "num_transform_blocks": 1,
            "activation": "elu",
            "dropout_probability": 0.0,
            "batch_norm": True,
            "num_bins": 4,
            "base_transform_type": "rq-coupling",
        },
    }


def cflow_posterior_kwargs():
    return {
        "input_dim": NUM_PARAMETERS,
        "context_dim": EMBEDDING_OUTPUT_DIM,
        "activation": "gelu",
        "batch_norm": False,
        "dropout": 0.0,
        "hidden_dims": [16, 16],
        "theta_with_glu": False,
        "context_with_glu": False,
        "time_prior_exponent": 1,
        "theta_embedding_kwargs": {
            "embedding_net": {
                "activation": "gelu",
                "hidden_dims": [8],
                "output_dim": 8,
                "type": "DenseResidualNet",
            },
            # NOTE: frequencies > 0 crashes on main: get_theta_embedding_net reads
            # `frequencies` from the top level of theta_embedding_kwargs while
            # get_dim_positional_embedding reads it from `encoding` — inconsistent.
            # The fmpe example only works because it uses frequencies: 0. Pinned here.
            "encoding": {"encode_all": False, "frequencies": 0},
        },
    }


def model_settings(posterior_model_type):
    if posterior_model_type == "normalizing_flow":
        posterior_kwargs = nsf_posterior_kwargs()
    else:
        posterior_kwargs = cflow_posterior_kwargs()
        if posterior_model_type == "flow_matching":
            posterior_kwargs["sigma_min"] = 0.001
        elif posterior_model_type == "score_matching":
            posterior_kwargs["epsilon"] = 1e-3
            posterior_kwargs["beta_min"] = 0.1
            posterior_kwargs["beta_max"] = 20.0
    return {
        "train_settings": {
            "model": {
                "posterior_model_type": posterior_model_type,
                "posterior_kwargs": posterior_kwargs,
                "embedding_kwargs": embedding_kwargs(),
            }
        }
    }


def data_sample(with_gnpe_proxies=False):
    """A sample in the format produced by the dataloader (wfd[0] after UnpackDict):
    [inference_parameters, strain data(, gnpe_proxies)]."""
    sample = [
        np.random.rand(NUM_PARAMETERS).astype(np.float32),
        np.random.rand(*DATA_SHAPE).astype(np.float32),
    ]
    if with_gnpe_proxies:
        sample.append(np.random.rand(GNPE_PROXY_DIM).astype(np.float32))
    return sample


def batch(model_type="normalizing_flow", with_gnpe_proxies=False):
    theta = torch.rand(BATCH_SIZE, NUM_PARAMETERS)
    context = [torch.rand(BATCH_SIZE, *DATA_SHAPE)]
    if with_gnpe_proxies:
        context.append(torch.rand(BATCH_SIZE, GNPE_PROXY_DIM))
    return theta, context


# -----------------------------------------------------------------------------------
# build_model_from_kwargs: type dispatch
# -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "posterior_model_type, expected_class",
    [
        ("normalizing_flow", NormalizingFlowPosteriorModel),
        ("flow_matching", FlowMatchingPosteriorModel),
        ("score_matching", ScoreDiffusionPosteriorModel),
    ],
)
def test_build_model_from_kwargs_dispatch(posterior_model_type, expected_class):
    settings = model_settings(posterior_model_type)
    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert type(pm) is expected_class
    assert isinstance(pm, BasePosteriorModel)
    assert pm.metadata is settings


def test_build_model_from_kwargs_rejects_unknown_type():
    settings = model_settings("normalizing_flow")
    settings["train_settings"]["model"]["posterior_model_type"] = "not_a_model"
    with pytest.raises(ValueError):
        build_model_from_kwargs(settings=settings, device="cpu")


def test_build_model_from_kwargs_requires_exactly_one_source():
    settings = model_settings("normalizing_flow")
    with pytest.raises(ValueError):
        build_model_from_kwargs(filename=None, settings=None)
    with pytest.raises(ValueError):
        build_model_from_kwargs(filename="model.pt", settings=settings)


# -----------------------------------------------------------------------------------
# autocomplete_model_kwargs: dimensional glue
# -----------------------------------------------------------------------------------


def test_autocomplete_model_kwargs_without_gnpe():
    model_kwargs = model_settings("normalizing_flow")["train_settings"]["model"]
    # Settings as written by a user: dims absent.
    del model_kwargs["embedding_kwargs"]["input_dims"]
    del model_kwargs["posterior_kwargs"]["input_dim"]
    del model_kwargs["posterior_kwargs"]["context_dim"]

    autocomplete_model_kwargs(model_kwargs, data_sample(with_gnpe_proxies=False))

    assert model_kwargs["embedding_kwargs"]["input_dims"] == list(DATA_SHAPE)
    assert model_kwargs["posterior_kwargs"]["input_dim"] == NUM_PARAMETERS
    assert model_kwargs["embedding_kwargs"]["added_context"] is False
    assert model_kwargs["posterior_kwargs"]["context_dim"] == EMBEDDING_OUTPUT_DIM


def test_autocomplete_model_kwargs_with_gnpe():
    model_kwargs = model_settings("normalizing_flow")["train_settings"]["model"]

    autocomplete_model_kwargs(model_kwargs, data_sample(with_gnpe_proxies=True))

    assert model_kwargs["embedding_kwargs"]["added_context"] is True
    assert (
        model_kwargs["posterior_kwargs"]["context_dim"]
        == EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM
    )


# -----------------------------------------------------------------------------------
# End-to-end: build + forward passes for all three model types
# -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "posterior_model_type", ["normalizing_flow", "flow_matching", "score_matching"]
)
def test_model_forward_passes(posterior_model_type):
    pm = build_model_from_kwargs(
        settings=model_settings(posterior_model_type), device="cpu"
    )
    theta, context = batch()

    loss = pm.loss(theta, *context)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    pm.network.eval()
    with torch.no_grad():
        samples = pm.sample(*context, num_samples=3)
        assert samples.shape == (BATCH_SIZE, 3, NUM_PARAMETERS)

        log_prob = pm.log_prob(theta, *context)
        assert log_prob.shape == (BATCH_SIZE,)
        assert torch.isfinite(log_prob).all()

        samples, log_prob = pm.sample_and_log_prob(*context, num_samples=3)
        assert samples.shape == (BATCH_SIZE, 3, NUM_PARAMETERS)
        assert log_prob.shape == (BATCH_SIZE, 3)


def test_normalizing_flow_with_gnpe_context():
    """With added_context=True, the embedding merges (data, proxies) via ModuleMerger,
    and sampling/density evaluation take two context tensors."""
    settings = model_settings("normalizing_flow")
    model = settings["train_settings"]["model"]
    model["embedding_kwargs"]["added_context"] = True
    model["posterior_kwargs"]["context_dim"] = EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM

    pm = build_model_from_kwargs(settings=settings, device="cpu")
    theta, context = batch(with_gnpe_proxies=True)

    loss = pm.loss(theta, *context)
    assert torch.isfinite(loss)

    pm.network.eval()
    with torch.no_grad():
        log_prob = pm.log_prob(theta, *context)
    assert log_prob.shape == (BATCH_SIZE,)


def test_normalizing_flow_unconditional():
    """Without embedding_kwargs, an unconditional flow is built (the models-as-priors
    path used by unconditional_density_estimation)."""
    settings = model_settings("normalizing_flow")
    model = settings["train_settings"]["model"]
    del model["embedding_kwargs"]
    # Convention from unconditional_density_estimation.py:74-75.
    model["posterior_kwargs"]["context_dim"] = None

    pm = build_model_from_kwargs(settings=settings, device="cpu")
    theta = torch.rand(BATCH_SIZE, NUM_PARAMETERS)

    loss = pm.loss(theta)
    assert torch.isfinite(loss)

    pm.network.eval()
    with torch.no_grad():
        samples = pm.sample(num_samples=3)
        assert samples.shape == (3, NUM_PARAMETERS)
        log_prob = pm.log_prob(theta)
        assert log_prob.shape == (BATCH_SIZE,)


# -----------------------------------------------------------------------------------
# SVD initial-weight seeding
# -----------------------------------------------------------------------------------


def test_initial_weights_seed_svd_projection():
    """initial_weights['V_rb_list'] seeds the LinearProjectionRB layer weights, and the
    settings dict is not polluted with the (large) V matrices."""
    num_bins = DATA_SHAPE[2]
    n_rb = 10
    V_rb_list = [
        (np.random.rand(num_bins, n_rb) + 1j * np.random.rand(num_bins, n_rb))
        for _ in range(DATA_SHAPE[0])
    ]
    settings = model_settings("normalizing_flow")
    settings_before = copy.deepcopy(settings)

    pm = build_model_from_kwargs(
        settings=settings, initial_weights={"V_rb_list": V_rb_list}, device="cpu"
    )

    projection = pm.network.embedding_net[0]
    V = V_rb_list[0][:, :n_rb]
    layer_weight = projection.layers_rb[0].weight.data
    assert torch.allclose(
        layer_weight[:n_rb, :num_bins],
        torch.from_numpy(V.real.T).float(),
    )
    assert torch.allclose(
        layer_weight[n_rb:, :num_bins],
        torch.from_numpy(V.imag.T).float(),
    )
    # The V matrices must not leak into the saved settings.
    assert settings == settings_before


# -----------------------------------------------------------------------------------
# Backward compatibility: old settings schema
# -----------------------------------------------------------------------------------


def test_update_model_config_maps_old_schema():
    old = {
        "type": "nsf+embedding",
        "nsf_kwargs": nsf_posterior_kwargs(),
        "embedding_net_kwargs": embedding_kwargs(),
    }
    update_model_config(old)
    assert old["posterior_model_type"] == "normalizing_flow"
    assert old["posterior_kwargs"] == nsf_posterior_kwargs()
    assert old["embedding_kwargs"] == embedding_kwargs()
    assert "type" not in old and "nsf_kwargs" not in old


def test_build_model_from_old_schema_settings():
    settings = model_settings("normalizing_flow")
    model = settings["train_settings"]["model"]
    settings["train_settings"]["model"] = {
        "type": "nsf+embedding",
        "nsf_kwargs": model["posterior_kwargs"],
        "embedding_net_kwargs": model["embedding_kwargs"],
    }
    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert type(pm) is NormalizingFlowPosteriorModel


# -----------------------------------------------------------------------------------
# Save / load round trip through build_model_from_kwargs (filename path)
# -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "posterior_model_type", ["normalizing_flow", "flow_matching"]
)
def test_save_and_rebuild_from_file(tmp_path, posterior_model_type):
    pm = build_model_from_kwargs(
        settings=model_settings(posterior_model_type), device="cpu"
    )
    filename = str(tmp_path / "model.pt")
    pm.save_model(filename)

    pm_loaded = build_model_from_kwargs(
        filename=filename, device="cpu", load_training_info=False
    )
    assert type(pm_loaded) is type(pm)
    for p0, p1 in zip(pm.network.parameters(), pm_loaded.network.parameters()):
        assert torch.equal(p0.data, p1.data)
