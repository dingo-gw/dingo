import numpy as np
import pytest

from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel


BASE_TRANSFORM_KWARGS = {
    "hidden_dim": 8,
    "num_transform_blocks": 1,
    "activation": "elu",
    "dropout_probability": 0.0,
    "batch_norm": False,
    "num_bins": 4,
    "base_transform_type": "rq-coupling",
}


def _settings(posterior_model_type="normalizing_flow"):
    return {
        "train_settings": {
            "model": {
                "posterior_model_type": posterior_model_type,
                "posterior_kwargs": {
                    "input_dim": 3,
                    "context_dim": None,
                    "num_flow_steps": 2,
                    "base_transform_kwargs": BASE_TRANSFORM_KWARGS,
                },
            }
        }
    }


def test_build_model_dispatches_to_normalizing_flow():
    model = build_model_from_kwargs(settings=_settings(), device="cpu")
    assert isinstance(model, NormalizingFlowPosteriorModel)


def test_build_model_dispatch_is_case_insensitive():
    model = build_model_from_kwargs(
        settings=_settings("Normalizing_Flow"), device="cpu"
    )
    assert isinstance(model, NormalizingFlowPosteriorModel)


def test_build_model_requires_exactly_one_of_filename_or_settings():
    # Neither provided.
    with pytest.raises(ValueError, match="filename or a settings"):
        build_model_from_kwargs()
    # Both provided.
    with pytest.raises(ValueError, match="filename or a settings"):
        build_model_from_kwargs(filename="x.pt", settings=_settings())


def test_build_model_rejects_unknown_type():
    with pytest.raises(ValueError, match="No valid posterior model type"):
        build_model_from_kwargs(settings=_settings("not_a_model"), device="cpu")


def test_autocomplete_model_kwargs_without_gnpe_proxies():
    model_kwargs = {"embedding_kwargs": {"output_dim": 8}, "posterior_kwargs": {}}
    # data_sample = [parameters, GW data]  (no gnpe proxies)
    autocomplete_model_kwargs(
        model_kwargs, data_sample=[np.zeros(4), np.zeros((2, 3, 20))]
    )

    assert model_kwargs["embedding_kwargs"]["input_dims"] == [2, 3, 20]
    assert model_kwargs["posterior_kwargs"]["input_dim"] == 4
    assert model_kwargs["embedding_kwargs"]["added_context"] is False
    # context_dim == embedding output_dim.
    assert model_kwargs["posterior_kwargs"]["context_dim"] == 8


def test_autocomplete_model_kwargs_with_gnpe_proxies():
    model_kwargs = {"embedding_kwargs": {"output_dim": 8}, "posterior_kwargs": {}}
    # data_sample = [parameters, GW data, gnpe_proxies (len 2)]
    autocomplete_model_kwargs(
        model_kwargs, data_sample=[np.zeros(4), np.zeros((2, 3, 20)), np.zeros(2)]
    )

    assert model_kwargs["embedding_kwargs"]["added_context"] is True
    # context_dim == output_dim + gnpe_proxy_dim == 8 + 2.
    assert model_kwargs["posterior_kwargs"]["context_dim"] == 10
