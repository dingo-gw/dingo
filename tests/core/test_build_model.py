"""
Tests for the model build path.

These tests cover building posterior models from settings dictionaries — type
dispatch (build_model_from_kwargs), per-architecture dimension inference
(complete_model_settings), end-to-end construction and forward passes for all three
posterior model types, SVD initial-weight seeding, and loading of old-schema
settings/checkpoints through the update_model_config boundary (see
hackathon/NN_Build_System_Design.md).
"""

import copy

import numpy as np
import pytest
import torch

from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.posterior_models.build_model import (
    build_model_from_kwargs,
    complete_model_settings,
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


def embedding_kwargs(completed=True):
    """Embedding-net kwargs; completed=False gives user-style settings (no dims)."""
    kwargs = {
        "svd": {"size": 10},
        "output_dim": EMBEDDING_OUTPUT_DIM,
        "hidden_dims": [32, 16, 8],
        "activation": "elu",
        "dropout": 0.0,
        "batch_norm": True,
    }
    if completed:
        kwargs["input_dims"] = list(DATA_SHAPE)
    return kwargs


def nsf_distribution_kwargs(completed=True):
    kwargs = {
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
    if completed:
        kwargs["theta_dim"] = NUM_PARAMETERS
        kwargs["context_dim"] = EMBEDDING_OUTPUT_DIM
    return kwargs


def cflow_distribution_kwargs(completed=True):
    kwargs = {
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
    if completed:
        kwargs["theta_dim"] = NUM_PARAMETERS
        kwargs["context_dim"] = EMBEDDING_OUTPUT_DIM
    return kwargs


def model_settings(model_type, completed=True):
    if model_type == "normalizing_flow":
        distribution_kwargs = nsf_distribution_kwargs(completed)
    else:
        distribution_kwargs = cflow_distribution_kwargs(completed)
        if model_type == "flow_matching":
            distribution_kwargs["sigma_min"] = 0.001
        elif model_type == "score_matching":
            distribution_kwargs["epsilon"] = 1e-3
            distribution_kwargs["beta_min"] = 0.1
            distribution_kwargs["beta_max"] = 20.0
    return {
        "train_settings": {
            "model": {
                "distribution": {"type": model_type, "kwargs": distribution_kwargs},
                "embedding_net": {
                    "type": "dense_svd",
                    "kwargs": embedding_kwargs(completed),
                },
            }
        }
    }


def data_sample(with_gnpe_proxies=False):
    """A sample in the format produced by the dataloader (wfd[0] after SelectKeys):
    a dict with inference_parameters, waveform(, context_parameters)."""
    sample = {
        "inference_parameters": np.random.rand(NUM_PARAMETERS).astype(np.float32),
        "waveform": np.random.rand(*DATA_SHAPE).astype(np.float32),
    }
    if with_gnpe_proxies:
        sample["context_parameters"] = np.random.rand(GNPE_PROXY_DIM).astype(np.float32)
    return sample


def batch(model_type="normalizing_flow", with_gnpe_proxies=False):
    theta = torch.rand(BATCH_SIZE, NUM_PARAMETERS)
    context = {"waveform": torch.rand(BATCH_SIZE, *DATA_SHAPE)}
    if with_gnpe_proxies:
        context["context_parameters"] = torch.rand(BATCH_SIZE, GNPE_PROXY_DIM)
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
    settings["train_settings"]["model"]["distribution"]["type"] = "not_a_model"
    with pytest.raises(ValueError):
        build_model_from_kwargs(settings=settings, device="cpu")


def test_build_model_from_kwargs_requires_exactly_one_source():
    settings = model_settings("normalizing_flow")
    with pytest.raises(ValueError):
        build_model_from_kwargs(filename=None, settings=None)
    with pytest.raises(ValueError):
        build_model_from_kwargs(filename="model.pt", settings=settings)


# -----------------------------------------------------------------------------------
# complete_model_settings: per-architecture dimension inference
# -----------------------------------------------------------------------------------


def test_complete_model_settings_without_context_parameters():
    user = model_settings("normalizing_flow", completed=False)["train_settings"][
        "model"
    ]
    completed = complete_model_settings(user, data_sample(with_gnpe_proxies=False))

    # The user settings are not modified; the completed ones carry the dims.
    assert "input_dims" not in user["embedding_net"]["kwargs"]
    assert completed["embedding_net"]["kwargs"]["input_dims"] == list(DATA_SHAPE)
    assert completed["distribution"]["kwargs"]["theta_dim"] == NUM_PARAMETERS
    assert completed["distribution"]["kwargs"]["context_dim"] == EMBEDDING_OUTPUT_DIM
    assert "context_merger" not in completed

    # The completed settings build directly.
    settings = {"train_settings": {"model": completed}}
    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert type(pm) is NormalizingFlowPosteriorModel


def test_complete_model_settings_with_context_parameters():
    """With context_parameters in the batch, a concat context merger is added and
    context_dim grows accordingly."""
    user = model_settings("normalizing_flow", completed=False)["train_settings"][
        "model"
    ]
    completed = complete_model_settings(user, data_sample(with_gnpe_proxies=True))

    assert completed["context_merger"] == {
        "type": "concat",
        "kwargs": {"num_context_parameters": GNPE_PROXY_DIM},
    }
    assert (
        completed["distribution"]["kwargs"]["context_dim"]
        == EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM
    )


def test_complete_model_settings_rejects_dims_in_user_settings():
    """Dimensions are derived from the data; specifying them is an error."""
    user = model_settings("normalizing_flow", completed=True)["train_settings"]["model"]
    with pytest.raises(ValueError, match="derived from the data"):
        complete_model_settings(user, data_sample())

    user = model_settings("normalizing_flow", completed=False)["train_settings"][
        "model"
    ]
    user["embedding_net"]["kwargs"]["input_dims"] = list(DATA_SHAPE)
    with pytest.raises(ValueError, match="derived from the data"):
        complete_model_settings(user, data_sample())


def test_complete_model_settings_rejects_unused_context_merger():
    """A context merger without context parameters in the data is an error, not
    silently dropped."""
    user = model_settings("normalizing_flow", completed=False)["train_settings"][
        "model"
    ]
    user["context_merger"] = {"type": "concat"}
    with pytest.raises(ValueError, match="context_merger"):
        complete_model_settings(user, data_sample(with_gnpe_proxies=False))


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

    loss = pm.loss(theta, context)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    pm.network.eval()
    with torch.no_grad():
        samples = pm.sample(context, num_samples=3)
        assert samples.shape == (BATCH_SIZE, 3, NUM_PARAMETERS)

        log_prob = pm.log_prob(theta, context)
        assert log_prob.shape == (BATCH_SIZE,)
        assert torch.isfinite(log_prob).all()

        samples, log_prob = pm.sample_and_log_prob(context, num_samples=3)
        assert samples.shape == (BATCH_SIZE, 3, NUM_PARAMETERS)
        assert log_prob.shape == (BATCH_SIZE, 3)


def test_normalizing_flow_with_gnpe_context():
    """With a concat context merger, the embedding merges (waveform,
    context_parameters), and sampling/density evaluation consume both context
    entries."""
    settings = model_settings("normalizing_flow")
    model = settings["train_settings"]["model"]
    model["context_merger"] = {
        "type": "concat",
        "kwargs": {"num_context_parameters": GNPE_PROXY_DIM},
    }
    model["distribution"]["kwargs"]["context_dim"] = (
        EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM
    )

    pm = build_model_from_kwargs(settings=settings, device="cpu")
    theta, context = batch(with_gnpe_proxies=True)

    loss = pm.loss(theta, context)
    assert torch.isfinite(loss)

    pm.network.eval()
    with torch.no_grad():
        log_prob = pm.log_prob(theta, context)
    assert log_prob.shape == (BATCH_SIZE,)

    # A missing declared context entry must fail loudly.
    with pytest.raises(ValueError, match="missing keys"):
        pm.log_prob(theta, {"waveform": context["waveform"]})


def test_normalizing_flow_unconditional():
    """Without an embedding_net, an unconditional flow is built (the
    models-as-priors path used by unconditional_density_estimation)."""
    settings = model_settings("normalizing_flow")
    model = settings["train_settings"]["model"]
    del model["embedding_net"]
    model["distribution"]["kwargs"]["context_dim"] = None

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
# Parameter-contract accessors (interface consumed by the factorized sampler)
# -----------------------------------------------------------------------------------


def test_parameter_contract_accessors():
    """inference_parameters / context_parameters / standardization read the model's
    own train_settings["data"], the interface FlowFactor.from_model consumes."""
    settings = model_settings("normalizing_flow")
    parameters = [f"p{i}" for i in range(NUM_PARAMETERS)]
    settings["train_settings"]["data"] = {
        "inference_parameters": parameters,
        "context_parameters": ["ra", "dec"],
        "standardization": {
            "mean": {p: 0.0 for p in parameters + ["ra", "dec"]},
            "std": {p: 1.0 for p in parameters + ["ra", "dec"]},
        },
    }
    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert pm.inference_parameters == parameters
    assert pm.context_parameters == ["ra", "dec"]
    assert set(pm.standardization["mean"]) == set(parameters + ["ra", "dec"])


def test_parameter_contract_defaults():
    """context_parameters is [] when absent (plain NPE) or None (written as null by
    some configs)."""
    settings = model_settings("normalizing_flow")
    settings["train_settings"]["data"] = {
        "inference_parameters": ["chirp_mass"],
        "standardization": {"mean": {"chirp_mass": 30.0}, "std": {"chirp_mass": 5.0}},
    }
    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert pm.context_parameters == []
    pm.metadata["train_settings"]["data"]["context_parameters"] = None
    assert pm.context_parameters == []


# -----------------------------------------------------------------------------------
# Data-driven weight initialization (init_data_spec / initialize_weights hooks)
# -----------------------------------------------------------------------------------


def _init_batches(waveform_len, batch_sizes):
    """Batches in the format the initialization dataloader provides: per-block
    complex strains plus parameters."""
    rng = np.random.default_rng(42)
    num_bins = DATA_SHAPE[2]
    for batch_size in batch_sizes:
        waveform = {}
        for block in ("H1", "L1"):
            strains = rng.normal(size=(batch_size, waveform_len)) + 1j * rng.normal(
                size=(batch_size, waveform_len)
            )
            # Entries outside the network's input range (leading excess) are zero.
            strains[:, : waveform_len - num_bins] = 0.0
            waveform[block] = strains
        yield {
            "waveform": waveform,
            "parameters": {"chirp_mass": rng.uniform(size=batch_size)},
        }


def test_svd_initialization_hook():
    """DenseSVDEmbedding requests clean, un-formatted data via init_data_spec and
    seeds its projection layer with per-block SVD bases from the provided batches;
    rows outside the network's input range are dropped."""
    n_rb = 10
    num_bins = DATA_SHAPE[2]
    waveform_len = num_bins + 5  # data longer than the network input
    settings = model_settings("normalizing_flow")
    embedding_kwargs = settings["train_settings"]["model"]["embedding_net"]["kwargs"]
    embedding_kwargs["svd"] = {"size": n_rb, "num_training_samples": 40}
    settings_before = copy.deepcopy(settings)

    pm = build_model_from_kwargs(settings=settings, device="cpu")
    embedding = pm.network.embedding_net

    spec = embedding.init_data_spec()
    assert spec == {
        "noise": False,
        "network_format": False,
        "fix_parameters": {"luminosity_distance": 100.0},
        "num_samples": 40,
    }

    # Two batches of 25 provide the 40 samples (iteration stops mid-batch).
    embedding.initialize_weights(_init_batches(waveform_len, [25, 25]))

    # The SVD itself is not deterministic (partial SVD with random start vector),
    # so check the structure: the weights hold an orthonormal complex basis V of
    # the network's input size, in the (real, imag) block layout of
    # LinearProjectionRB.init_layers, with zero bias.
    for layer in embedding[0].layers_rb:
        weight = layer.weight.data
        V_real = weight[:n_rb, :num_bins].T
        V_imag = weight[n_rb:, :num_bins].T
        assert torch.allclose(weight[:n_rb, num_bins : 2 * num_bins].T, -V_imag)
        assert torch.allclose(weight[n_rb:, num_bins : 2 * num_bins].T, V_real)
        # Third channel (e.g. ASD) initialized to zero.
        assert torch.all(weight[:, 2 * num_bins :] == 0)
        assert torch.all(layer.bias.data == 0)
        V = torch.complex(V_real, V_imag).to(torch.complex128)
        gram = V.T.conj() @ V
        assert torch.allclose(gram, torch.eye(n_rb, dtype=torch.complex128), atol=1e-5)
    # The two blocks received different bases (different data).
    assert not torch.allclose(
        embedding[0].layers_rb[0].weight, embedding[0].layers_rb[1].weight
    )
    # The V matrices must not leak into the saved settings.
    assert settings == settings_before

    # Too few samples fail loudly.
    with pytest.raises(IndexError, match="40"):
        embedding.initialize_weights(_init_batches(waveform_len, [25]))


def test_init_data_spec_none_without_seeding_request():
    """Without num_training_samples in the svd settings (e.g. when loading a saved
    model), no data-driven initialization is requested."""
    pm = build_model_from_kwargs(
        settings=model_settings("normalizing_flow"), device="cpu"
    )
    assert pm.network.embedding_net.init_data_spec() is None


# -----------------------------------------------------------------------------------
# Backward compatibility: old settings schema
# -----------------------------------------------------------------------------------


def old_schema_model(model_type="normalizing_flow", added_context=False):
    """A completed model config in the old schema, as found in old checkpoints."""
    if model_type == "normalizing_flow":
        posterior_kwargs = nsf_distribution_kwargs()
    else:
        posterior_kwargs = cflow_distribution_kwargs()
        posterior_kwargs["sigma_min"] = 0.001
    posterior_kwargs["input_dim"] = posterior_kwargs.pop("theta_dim")
    old_embedding_kwargs = embedding_kwargs()
    old_embedding_kwargs["V_rb_list"] = None
    old_embedding_kwargs["added_context"] = added_context
    if added_context:
        posterior_kwargs["context_dim"] = EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM
    return {
        "posterior_model_type": model_type,
        "posterior_kwargs": posterior_kwargs,
        "embedding_kwargs": old_embedding_kwargs,
    }


def test_update_model_config_maps_old_schemas():
    """All old schemas map forward to the current one, including the oldest
    nsf+embedding form; the mapping is idempotent."""
    old = old_schema_model(added_context=True)
    oldest = {
        "type": "nsf+embedding",
        "nsf_kwargs": old["posterior_kwargs"],
        "embedding_net_kwargs": old["embedding_kwargs"],
    }
    for settings in (old, oldest):
        update_model_config(settings)
        assert settings["distribution"]["type"] == "normalizing_flow"
        assert settings["distribution"]["kwargs"]["theta_dim"] == NUM_PARAMETERS
        assert settings["embedding_net"]["type"] == "dense_svd"
        kwargs = settings["embedding_net"]["kwargs"]
        assert "added_context" not in kwargs and "V_rb_list" not in kwargs
        # Old concatenated context maps to the concat merger, with the number of
        # context parameters recovered from the completed dims.
        assert settings["context_merger"] == {
            "type": "concat",
            "kwargs": {"num_context_parameters": GNPE_PROXY_DIM},
        }
        before = copy.deepcopy(settings)
        update_model_config(settings)
        assert settings == before


def test_update_model_config_lowercases_builtin_type_names():
    """Model types used to be matched case-insensitively; the compat shim lowercases
    built-in names from old checkpoints so the case-sensitive registry finds them."""
    model = old_schema_model()
    model["posterior_model_type"] = "Normalizing_Flow"
    settings = {"train_settings": {"model": model}}
    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert type(pm) is NormalizingFlowPosteriorModel


@pytest.mark.parametrize("added_context", [False, True])
def test_old_schema_builds_state_dict_compatible_network(added_context):
    """A network built from old-schema settings has exactly the same state-dict keys
    and shapes as one built from the new schema — old checkpoints stay loadable."""
    old_settings = {
        "train_settings": {"model": old_schema_model(added_context=added_context)}
    }
    pm_old = build_model_from_kwargs(settings=old_settings, device="cpu")

    new_settings = model_settings("normalizing_flow")
    if added_context:
        model = new_settings["train_settings"]["model"]
        model["context_merger"] = {
            "type": "concat",
            "kwargs": {"num_context_parameters": GNPE_PROXY_DIM},
        }
        model["distribution"]["kwargs"]["context_dim"] = (
            EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM
        )
    pm_new = build_model_from_kwargs(settings=new_settings, device="cpu")

    state_old = pm_old.network.state_dict()
    state_new = pm_new.network.state_dict()
    assert list(state_old) == list(state_new)
    assert all(state_old[k].shape == state_new[k].shape for k in state_old)


@pytest.mark.parametrize("model_type", ["normalizing_flow", "flow_matching"])
def test_load_old_schema_checkpoint_file(tmp_path, model_type):
    """A checkpoint file whose model_kwargs use the old schema loads through the
    update_model_config boundary, with identical weights."""
    pm = build_model_from_kwargs(settings=model_settings(model_type), device="cpu")

    old_model = old_schema_model(model_type)
    checkpoint = {
        "model_kwargs": old_model,
        "model_state_dict": pm.network.state_dict(),
        "epoch": 3,
        "version": "dingo=0.9.9",
        "metadata": {"train_settings": {"model": copy.deepcopy(old_model)}},
    }
    filename = str(tmp_path / "old_model.pt")
    torch.save(checkpoint, filename)

    pm_loaded = build_model_from_kwargs(
        filename=filename, device="cpu", load_training_info=False
    )
    assert type(pm_loaded) is type(pm)
    assert pm_loaded.epoch == 3
    for p0, p1 in zip(pm.network.parameters(), pm_loaded.network.parameters()):
        assert torch.equal(p0.data, p1.data)


# -----------------------------------------------------------------------------------
# Save / load round trip through build_model_from_kwargs (filename path)
# -----------------------------------------------------------------------------------


@pytest.mark.parametrize("posterior_model_type", ["normalizing_flow", "flow_matching"])
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


# -----------------------------------------------------------------------------------
# Transformer embedding through the generic build path
# -----------------------------------------------------------------------------------

NUM_TOKENS = 6
NUM_FEATURES = 12
NUM_BLOCKS = 2


def transformer_embedding_settings():
    return {
        "type": "transformer",
        "kwargs": {
            "tokenizer_kwargs": {
                "hidden_dims": [16],
                "activation": "elu",
                "batch_norm": False,
                "layer_norm": True,
            },
            "transformer_kwargs": {
                "d_model": 16,
                "dim_feedforward": 32,
                "nhead": 4,
                "dropout": 0.0,
                "num_layers": 1,
                "norm_first": True,
            },
            "pooling": "cls",
            "final_net_kwargs": {
                "activation": "elu",
                "output_dim": EMBEDDING_OUTPUT_DIM,
            },
        },
    }


def tokenized_data_sample(with_context_parameters=False):
    position = np.stack(
        [
            np.linspace(20.0, 100.0, NUM_TOKENS),
            np.linspace(30.0, 110.0, NUM_TOKENS),
            np.repeat(np.arange(NUM_BLOCKS), NUM_TOKENS // NUM_BLOCKS),
        ],
        axis=-1,
    ).astype(np.float32)
    sample = {
        "inference_parameters": np.random.rand(NUM_PARAMETERS).astype(np.float32),
        "waveform": np.random.rand(NUM_TOKENS, NUM_FEATURES).astype(np.float32),
        "position": position,
        "drop_token_mask": np.zeros(NUM_TOKENS, dtype=bool),
    }
    if with_context_parameters:
        sample["context_parameters"] = np.random.rand(GNPE_PROXY_DIM).astype(np.float32)
    return sample


def tokenized_batch(with_context_parameters=False):
    sample = tokenized_data_sample(with_context_parameters)
    context = {
        k: torch.from_numpy(np.stack([v] * BATCH_SIZE))
        for k, v in sample.items()
        if k != "inference_parameters"
    }
    theta = torch.rand(BATCH_SIZE, NUM_PARAMETERS)
    return theta, context


@pytest.mark.parametrize("model_type", ["normalizing_flow", "flow_matching"])
def test_transformer_embedding_composes_with_any_distribution(model_type):
    """The transformer works with any registered distribution type through the
    generic build path — including flow matching, which the original transformer
    branch never wired up."""
    settings = model_settings(model_type, completed=False)
    model = settings["train_settings"]["model"]
    model["embedding_net"] = transformer_embedding_settings()

    model = complete_model_settings(model, tokenized_data_sample())
    assert model["embedding_net"]["kwargs"]["tokenizer_kwargs"]["num_blocks"] == (
        NUM_BLOCKS
    )
    assert model["distribution"]["kwargs"]["context_dim"] == EMBEDDING_OUTPUT_DIM

    pm = build_model_from_kwargs(
        settings={"train_settings": {"model": model}}, device="cpu"
    )
    theta, context = tokenized_batch()

    loss = pm.loss(theta, context)
    assert torch.isfinite(loss)

    pm.network.eval()
    with torch.no_grad():
        samples = pm.sample(context, num_samples=2)
    assert samples.shape == (BATCH_SIZE, 2, NUM_PARAMETERS)


def test_transformer_embedding_with_context_parameters():
    """Context parameters compose with the transformer via the generic concat
    merger — on the original branch this was impossible (the batch slot for
    proxies was occupied by the position tensor)."""
    settings = model_settings("normalizing_flow", completed=False)
    model = settings["train_settings"]["model"]
    model["embedding_net"] = transformer_embedding_settings()

    model = complete_model_settings(
        model, tokenized_data_sample(with_context_parameters=True)
    )
    assert model["context_merger"]["kwargs"]["num_context_parameters"] == (
        GNPE_PROXY_DIM
    )
    assert model["distribution"]["kwargs"]["context_dim"] == (
        EMBEDDING_OUTPUT_DIM + GNPE_PROXY_DIM
    )

    pm = build_model_from_kwargs(
        settings={"train_settings": {"model": model}}, device="cpu"
    )
    assert pm.network.context_keys == (
        "waveform",
        "position",
        "drop_token_mask",
        "context_parameters",
    )
    theta, context = tokenized_batch(with_context_parameters=True)
    assert torch.isfinite(pm.loss(theta, context))
