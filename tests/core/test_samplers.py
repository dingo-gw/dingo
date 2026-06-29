import numpy as np
import pandas as pd
import pytest
import torch

from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.result import Result
from dingo.core.samplers import Sampler


INFERENCE_PARAMETERS = ["a", "b", "c"]

# Non-trivial standardization so that the log_prob change-of-variables correction
# (-sum(log(std))) is actually exercised by the round-trip test below.
STANDARDIZATION = {
    "mean": {"a": 1.0, "b": -2.0, "c": 0.5},
    "std": {"a": 2.0, "b": 0.5, "c": 3.0},
}

BASE_TRANSFORM_KWARGS = {
    "hidden_dim": 16,
    "num_transform_blocks": 1,
    "activation": "elu",
    "dropout_probability": 0.0,
    "batch_norm": False,
    "num_bins": 4,
    "base_transform_type": "rq-coupling",
}


def _build_model(unconditional, context_dim=None, embedding_kwargs=None):
    """Build a small normalizing-flow posterior model for use with a Sampler."""
    posterior_kwargs = {
        "input_dim": len(INFERENCE_PARAMETERS),
        "context_dim": context_dim,
        "num_flow_steps": 2,
        "base_transform_kwargs": BASE_TRANSFORM_KWARGS,
    }
    model_kwargs = {
        "posterior_model_type": "normalizing_flow",
        "posterior_kwargs": posterior_kwargs,
    }
    if embedding_kwargs is not None:
        model_kwargs["embedding_kwargs"] = embedding_kwargs

    metadata = {
        "train_settings": {
            "model": model_kwargs,
            "data": {
                "inference_parameters": INFERENCE_PARAMETERS,
                "standardization": STANDARDIZATION,
                "unconditional": unconditional,
            },
        }
    }
    if unconditional:
        # For unconditional models the Sampler keeps the precursor ("base") metadata
        # separately; it just needs to exist.
        metadata["base"] = {}

    return NormalizingFlowPosteriorModel(metadata=metadata, device="cpu")


@pytest.fixture()
def unconditional_sampler():
    """Sampler wrapping an unconditional flow.

    This takes the context-free (x = []) path through run_sampler / log_prob, which
    isolates the count, batching, and log_prob logic with minimal setup.
    """
    return Sampler(model=_build_model(unconditional=True))


@pytest.fixture()
def conditional_sampler():
    """Sampler wrapping a conditional flow, without an embedding network.

    Sufficient for exercising the context-required error paths, which are checked
    before the network is ever called.
    """
    return Sampler(model=_build_model(unconditional=False, context_dim=5))


@pytest.fixture()
def conditional_sampler_with_context():
    """Sampler wrapping a conditional flow *with* an embedding network.

    The full conditional sampling path runs here. The base Sampler's default
    transform_pre is the identity (a subclass would replace it); we substitute a
    minimal callable that extracts the context tensor so that data reaches the
    embedding network.
    """
    embedding_kwargs = {
        "input_dims": (2, 3, 20),
        "svd": {"size": 10},
        "V_rb_list": None,
        "output_dim": 8,
        "hidden_dims": [32, 16, 8],
        "activation": "elu",
        "dropout": 0.0,
        "batch_norm": False,
        "added_context": False,
    }
    sampler = Sampler(
        model=_build_model(
            unconditional=False, context_dim=8, embedding_kwargs=embedding_kwargs
        )
    )
    sampler.transform_pre = lambda context: context["data"]
    return sampler


def test_run_sampler_returns_correct_number_of_samples(unconditional_sampler):
    unconditional_sampler.run_sampler(num_samples=17)
    assert len(unconditional_sampler.samples) == 17


def test_samples_columns_are_inference_parameters_plus_log_prob(unconditional_sampler):
    unconditional_sampler.run_sampler(num_samples=10)
    assert list(unconditional_sampler.samples.columns) == INFERENCE_PARAMETERS + [
        "log_prob"
    ]


@pytest.mark.parametrize("batch_size", [None, 3, 5, 7, 10, 25])
def test_run_sampler_count_invariant_to_batch_size(unconditional_sampler, batch_size):
    """The number of samples must not depend on batch_size.

    Covers the divmod branch in run_sampler: exact divisors (5, 10), divisors with a
    remainder (3, 7), a batch larger than the request (25 -> one short batch), and the
    default single-batch case (None).
    """
    num_samples = 10
    unconditional_sampler.run_sampler(num_samples=num_samples, batch_size=batch_size)
    assert len(unconditional_sampler.samples) == num_samples


def test_conditional_run_sampler_without_context_raises(conditional_sampler):
    with pytest.raises(ValueError, match="Context must be set"):
        conditional_sampler.run_sampler(num_samples=5)


def test_conditional_log_prob_without_context_raises(conditional_sampler):
    samples = pd.DataFrame({p: [0.0] for p in INFERENCE_PARAMETERS})
    with pytest.raises(ValueError, match="Context must be set"):
        conditional_sampler.log_prob(samples)


def test_unconditional_run_sampler_needs_no_context(unconditional_sampler):
    # Unconditional models carry no context and require none in order to sample.
    assert unconditional_sampler.context is None
    unconditional_sampler.run_sampler(num_samples=8)
    assert len(unconditional_sampler.samples) == 8


def test_log_prob_round_trip_matches_sampling_log_prob(unconditional_sampler):
    """log_prob recomputed at the sampled points reproduces the sampling log_prob.

    This exercises the full standardize -> de-standardize round trip, including the
    change-of-variables correction (-sum(log(std))). The stored log_prob column must
    be dropped first, since log_prob() adds any log_prob already present (see
    test_log_prob_adds_existing_log_prob_column).
    """
    unconditional_sampler.run_sampler(num_samples=20)
    samples = unconditional_sampler.samples
    recomputed = unconditional_sampler.log_prob(samples.drop(columns="log_prob"))
    np.testing.assert_allclose(
        recomputed, samples["log_prob"].to_numpy(), atol=1e-4
    )


def test_log_prob_adds_existing_log_prob_column(unconditional_sampler):
    """A log_prob column in the input is added to the recomputed network log_prob.

    This is relied upon by GNPE / density-recovery, where the carried log_prob is a
    proposal contribution. The base Sampler does not drop it.
    """
    unconditional_sampler.run_sampler(num_samples=6)
    samples = unconditional_sampler.samples
    stored = samples["log_prob"].to_numpy()
    recomputed = unconditional_sampler.log_prob(samples.drop(columns="log_prob"))
    combined = unconditional_sampler.log_prob(samples)
    np.testing.assert_allclose(combined, recomputed + stored, atol=1e-4)


def test_log_prob_accepts_dataframe_and_dict_inputs(unconditional_sampler):
    unconditional_sampler.run_sampler(num_samples=6)
    samples = unconditional_sampler.samples.drop(columns="log_prob")

    # DataFrame -> one log_prob per row.
    lp_dataframe = unconditional_sampler.log_prob(samples)
    assert lp_dataframe.shape == (6,)

    # dict of arrays -> one log_prob per row.
    lp_dict = unconditional_sampler.log_prob(
        {p: samples[p].to_numpy() for p in INFERENCE_PARAMETERS}
    )
    assert lp_dict.shape == (6,)

    # dict of scalars -> a single log_prob.
    lp_scalar = unconditional_sampler.log_prob(
        {p: samples[p].iloc[0] for p in INFERENCE_PARAMETERS}
    )
    assert lp_scalar.shape == (1,)


def test_to_result_round_trips_samples_and_settings(unconditional_sampler):
    unconditional_sampler.run_sampler(num_samples=10)
    result = unconditional_sampler.to_result()
    assert isinstance(result, Result)
    pd.testing.assert_frame_equal(result.samples, unconditional_sampler.samples)
    assert result.settings == unconditional_sampler.metadata


def test_to_hdf5_round_trips_samples(unconditional_sampler, tmp_path):
    unconditional_sampler.run_sampler(num_samples=10)
    unconditional_sampler.to_hdf5(label="result", outdir=str(tmp_path))

    reloaded = Result(file_name=str(tmp_path / "result.hdf5"))
    np.testing.assert_allclose(
        reloaded.samples[INFERENCE_PARAMETERS].to_numpy(),
        unconditional_sampler.samples[INFERENCE_PARAMETERS].to_numpy(),
    )


def test_context_setter_moves_parameters_to_event_metadata(conditional_sampler):
    conditional_sampler.context = {"waveform": [1, 2, 3], "parameters": {"mass": 30.0}}
    assert conditional_sampler.event_metadata["injection_parameters"] == {"mass": 30.0}
    # The injected truth parameters are removed from the context itself.
    assert "parameters" not in conditional_sampler.context


def test_conditional_samples_depend_on_context(conditional_sampler_with_context):
    """For a conditional model the samples depend on the context.

    With the torch RNG fixed, the same context reproduces samples exactly, while a
    different context changes them -- isolating the context as the source of variation.
    """
    sampler = conditional_sampler_with_context
    context_a = {"data": torch.randn(2, 3, 20)}
    context_b = {"data": torch.randn(2, 3, 20)}

    torch.manual_seed(0)
    sampler.context = context_a
    sampler.run_sampler(num_samples=8)
    samples_a = sampler.samples.to_numpy().copy()

    torch.manual_seed(0)
    sampler.context = context_a
    sampler.run_sampler(num_samples=8)
    samples_a_repeat = sampler.samples.to_numpy().copy()

    torch.manual_seed(0)
    sampler.context = context_b
    sampler.run_sampler(num_samples=8)
    samples_b = sampler.samples.to_numpy().copy()

    assert np.array_equal(samples_a, samples_a_repeat)
    assert not np.array_equal(samples_a, samples_b)
