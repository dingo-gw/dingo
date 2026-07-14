"""Unit tests for the generic ``ComposedSampler`` runner and for ``FlowFactor`` driving
a *real* normalizing flow.

``test_factors`` exercises ``ChainComposer`` and the batching plumbing with network-free
mock factors; this file covers the two pieces that require an actual network:

* ``ComposedSampler.run_sampler`` -- the DataFrame runner over a chain (sample count,
  column layout, and batch-size invariance).
* ``FlowFactor`` around a genuine ``NormalizingFlowPosteriorModel`` -- the real
  standardize/de-standardize round trip, including the change-of-variables correction
  (``-sum(log std)``), which the deterministic fake model in ``test_factors`` cannot
  reach. Covered in both conditioning shapes: the *unconditional* flow (context-free,
  ``context=None``) and the *data-conditional* flow, whose data enters through a stub
  ``SamplerContext`` serving a fixed random tensor as ``prepared_data()`` -- no trained
  model or GW data machinery is needed for either.

(Adapts the runner/log_prob coverage from the pre-sampler-revamp ``test_samplers.py``,
which imported the now-deleted ``dingo.core.samplers.Sampler``.)
"""

import numpy as np
import pytest
import torch

from dingo.core.factors import ChainComposer, ComposedSampler, FlowFactor
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel


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


# Network-input shape served by the fake context below; the embedding network is
# sized to consume exactly this.
DATA_SHAPE = (2, 3, 20)

EMBEDDING_KWARGS = {
    "input_dims": DATA_SHAPE,
    "svd": {"size": 10},
    "V_rb_list": None,
    "output_dim": 8,
    "hidden_dims": [32, 16, 8],
    "activation": "elu",
    "dropout": 0.0,
    "batch_norm": False,
    "added_context": False,
}


def _build_model(unconditional, context_dim=None, embedding_kwargs=None):
    """A small (real but untrained) normalizing-flow posterior model."""
    model_kwargs = {
        "posterior_model_type": "normalizing_flow",
        "posterior_kwargs": {
            "input_dim": len(INFERENCE_PARAMETERS),
            "context_dim": context_dim,
            "num_flow_steps": 2,
            "base_transform_kwargs": BASE_TRANSFORM_KWARGS,
        },
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
        },
    }
    if unconditional:
        # Unconditional models keep the precursor ("base") metadata separately; the
        # FlowFactor must read the model's *own* standardization, not this.
        metadata["base"] = {}
    return NormalizingFlowPosteriorModel(metadata=metadata, device="cpu")


def _build_unconditional_model():
    return _build_model(unconditional=True)


def _build_conditional_model():
    """A data-conditional flow with an embedding network sized to ``DATA_SHAPE``."""
    return _build_model(
        unconditional=False,
        context_dim=EMBEDDING_KWARGS["output_dim"],
        embedding_kwargs=EMBEDDING_KWARGS,
    )


class _FakeDataContext:
    """Minimal ``SamplerContext`` stub for the data-conditional ``FlowFactor`` branch:
    ``prepared_data()`` serves a stored, *unbatched* network-input tensor (the factor
    adds the batch dimension itself: ``unsqueeze(0)`` at sample time, ``expand`` in
    ``log_prob``)."""

    def __init__(self, data):
        self._data = data

    def prepared_data(self, conditioning=None):
        assert conditioning is None  # plain NPE: no chain columns enter the data prep
        return self._data


@pytest.fixture()
def sampler():
    """A ``ComposedSampler`` over a single unconditional ``FlowFactor`` (context-free)."""
    factor = FlowFactor.from_model(_build_unconditional_model())
    return ComposedSampler(ChainComposer([factor]), context=None)


def test_run_sampler_returns_correct_number_of_samples(sampler):
    sampler.run_sampler(num_samples=17)
    assert len(sampler.samples) == 17


def test_samples_columns_are_inference_parameters_plus_log_prob(sampler):
    sampler.run_sampler(num_samples=10)
    assert list(sampler.samples.columns) == INFERENCE_PARAMETERS + ["log_prob"]


@pytest.mark.parametrize("batch_size", [None, 3, 5, 7, 10, 25])
def test_run_sampler_count_invariant_to_batch_size(sampler, batch_size):
    """The number of samples must not depend on batch_size.

    Covers the chunking in ComposedSampler.run_sampler / chunk_and_concat: exact
    divisors (5, 10), divisors with a remainder (3, 7), a batch larger than the request
    (25 -> one short batch), and the default single-batch case (None).
    """
    num_samples = 10
    sampler.run_sampler(num_samples=num_samples, batch_size=batch_size)
    assert len(sampler.samples) == num_samples


def test_flow_factor_log_prob_round_trip_matches_sampling_log_prob():
    """log_prob recomputed at the sampled points reproduces the sampling log_prob.

    This exercises the full standardize -> de-standardize round trip through a real
    flow, including the change-of-variables correction (-sum(log(std))). The fake model
    used in ``test_factors`` is deterministic and cannot reach this path.
    """
    factor = FlowFactor.from_model(_build_unconditional_model())
    torch.manual_seed(0)
    samples, log_prob = factor.sample_and_log_prob(20, context=None)
    recomputed = factor.log_prob(samples, context=None)
    # The network runs in float32, and sampling vs. log_prob use the flow's forward
    # vs. inverse transforms, which round-trip only to the float32 floor (observed
    # ~1e-6); atol=1e-5 leaves ~10x margin.
    np.testing.assert_allclose(
        recomputed.numpy(), log_prob.numpy(), atol=1e-5
    )


def test_conditional_samples_depend_on_context():
    """For a data-conditional model the samples depend on the context data.

    With the torch RNG fixed, the same context reproduces samples exactly, while a
    different context changes them -- isolating ``prepared_data()`` as the source of
    variation. This is the ``FlowFactor`` branch that draws from the shared data
    context (an untrained flow + embedding network suffices).
    """
    factor = FlowFactor.from_model(_build_conditional_model())
    context_a = _FakeDataContext(torch.randn(*DATA_SHAPE))
    context_b = _FakeDataContext(torch.randn(*DATA_SHAPE))

    torch.manual_seed(0)
    samples_a, _ = factor.sample_and_log_prob(8, context_a)
    torch.manual_seed(0)
    samples_a_repeat, _ = factor.sample_and_log_prob(8, context_a)
    torch.manual_seed(0)
    samples_b, _ = factor.sample_and_log_prob(8, context_b)

    assert all(samples_a[p].shape == (8,) for p in INFERENCE_PARAMETERS)
    # Same seed + same context -> identical samples.
    assert all(
        torch.equal(samples_a[p], samples_a_repeat[p]) for p in INFERENCE_PARAMETERS
    )
    # Same seed + different context -> different samples.
    assert not all(
        torch.equal(samples_a[p], samples_b[p]) for p in INFERENCE_PARAMETERS
    )


def test_conditional_log_prob_round_trip_matches_sampling_log_prob():
    """log_prob recomputed at the sampled points reproduces the sampling log_prob,
    through the data-conditional branch.

    Sampling conditions on ``prepared_data().unsqueeze(0)`` (one shared context row,
    squeezed back out); ``log_prob`` instead expands the data across the sample rows
    (``data.expand(num_samples, ...)``) -- the only CI coverage of that expand path.
    """
    factor = FlowFactor.from_model(_build_conditional_model())
    context = _FakeDataContext(torch.randn(*DATA_SHAPE))
    torch.manual_seed(0)
    samples, log_prob = factor.sample_and_log_prob(20, context)
    recomputed = factor.log_prob(samples, context)
    # float32 round-trip floor, as in the unconditional round-trip test above.
    np.testing.assert_allclose(recomputed.numpy(), log_prob.numpy(), atol=1e-5)
