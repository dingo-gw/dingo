"""
Unit tests for ``GWSamplerContext`` plumbing: the likelihood cache, derived
contexts (importance-sampling representations), the device attribute, and the
metadata constructors. The likelihood itself is stubbed (no waveform generation);
the model-based construction parity lives in the local harnesses.
"""

import copy

import numpy as np
import pytest
import torch

import dingo.gw.inference.context as context_module
from dingo.gw.domains import build_domain
from dingo.gw.inference.context import GWSamplerContext

_DOMAIN_SETTINGS = {
    "type": "FrequencyDomain",
    "f_min": 20.0,
    "f_max": 1024.0,
    "delta_f": 0.25,
}

_BASE_METADATA = {
    "dataset_settings": {
        "domain": _DOMAIN_SETTINGS,
        "waveform_generator": {"approximant": "IMRPhenomD", "f_ref": 20.0},
    },
    "train_settings": {
        "data": {
            "inference_parameters": ["chirp_mass"],
            "detectors": ["H1"],
            "ref_time": 1126259462.4,
        }
    },
}


class _StubLikelihood:
    """Records constructions; stands in for StationaryGaussianGWLikelihood."""

    constructions = 0

    def __init__(self, **kwargs):
        type(self).constructions += 1
        self.kwargs = kwargs


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setattr(
        context_module, "StationaryGaussianGWLikelihood", _StubLikelihood
    )
    _StubLikelihood.constructions = 0
    return GWSamplerContext(
        domain=build_domain(_DOMAIN_SETTINGS),
        data_prep=None,
        event_data={},
        model_metadata=_BASE_METADATA,
    )


def test_likelihood_rebuilt_only_when_settings_change(context):
    # Repeated calls with unchanged arguments share the last-built instance (the
    # synthetic-phase factor requests one per chain chunk); a settings change
    # builds a replacement.
    default = context.likelihood()
    assert context.likelihood() is default
    assert _StubLikelihood.constructions == 1

    marg_kwargs = {"num_calibration_curves": 100}
    marginalized = context.likelihood(calibration_marginalization_kwargs=marg_kwargs)
    assert marginalized is not default
    assert context.likelihood(calibration_marginalization_kwargs=marg_kwargs) is (
        marginalized
    )
    assert _StubLikelihood.constructions == 2


def test_domain_eq_is_none_safe():
    domain = build_domain(_DOMAIN_SETTINGS)
    assert domain == build_domain(_DOMAIN_SETTINGS)
    assert domain != None  # noqa: E711
    assert (domain == 3) is False


def test_derived_context_has_independent_likelihood_cache(context):
    # The representation is context state: a derived context builds its own
    # likelihood (fresh cache) without disturbing the original's.
    default = context.likelihood()
    derived = context.derive(use_base_domain=True)
    base = derived.likelihood()
    assert base is not default
    assert base.kwargs["use_base_domain"] is True
    assert derived.likelihood() is base
    assert context.likelihood() is default
    assert context.use_base_domain is False


def test_derive_with_updated_duration_rebuilds_domain(context):
    # An updated duration T rebuilds the data domain at delta_f = 1/T and enters
    # waveform generation as wfg_delta_f; the event payload and metadata are
    # shared with the original context.
    derived = context.derive(updates={"T": 2.0})
    assert derived.domain.delta_f == 0.5
    assert derived.wfg_delta_f == 0.5
    assert derived.event_data is context.event_data
    assert derived.model_metadata is context.model_metadata
    assert derived.t_ref == context.t_ref
    likelihood = derived.likelihood()
    assert likelihood.kwargs["data_domain"] is derived.domain
    assert likelihood.kwargs["wfg_domain"].delta_f == 0.5
    # The original context is untouched.
    assert context.domain.delta_f == _DOMAIN_SETTINGS["delta_f"]
    assert context.wfg_delta_f is None
    assert context.likelihood().kwargs["wfg_domain"].delta_f == (
        _DOMAIN_SETTINGS["delta_f"]
    )


def test_derive_with_frequency_updates_keeps_domain_grid(context):
    # min/max frequency updates apply via ASD masking (through the event
    # metadata), not the domain grid: the rebuild triggers but reproduces the
    # same domain settings.
    derived = context.derive(updates={"minimum_frequency": 25.0})
    assert derived.domain is not context.domain
    assert derived.domain.domain_dict == context.domain.domain_dict


def test_likelihood_frequency_range_follows_event_metadata(context):
    # The likelihood masks the ASDs to the event's frequency range; without an
    # event override the range defaults to the domain bounds.
    default_update = context.likelihood().kwargs["frequency_update"]
    assert default_update == {
        "minimum_frequency": _DOMAIN_SETTINGS["f_min"],
        "maximum_frequency": _DOMAIN_SETTINGS["f_max"],
    }
    with_event = GWSamplerContext(
        domain=build_domain(_DOMAIN_SETTINGS),
        data_prep=None,
        event_data={},
        event_metadata={"minimum_frequency": 21.0, "maximum_frequency": 512.0},
        model_metadata=_BASE_METADATA,
    )
    update = with_event.likelihood().kwargs["frequency_update"]
    assert update == {"minimum_frequency": 21.0, "maximum_frequency": 512.0}


def _event_data(n_bins):
    strain = np.ones(n_bins, dtype=complex)
    asd = np.ones(n_bins)
    return {
        "waveform": {"H1": strain.copy(), "L1": strain.copy()},
        "asds": {"H1": asd.copy(), "L1": asd.copy()},
    }


def _crop_context(event_metadata, crop_settings=None):
    meta = copy.deepcopy(_MODEL_METADATA)
    if crop_settings is not None:
        meta["train_settings"]["data"]["random_strain_cropping"] = crop_settings
    n_bins = int(_DOMAIN_SETTINGS["f_max"] / _DOMAIN_SETTINGS["delta_f"]) + 1
    return GWSamplerContext.from_model_metadata(
        meta, _event_data(n_bins), event_metadata=event_metadata
    )


def test_frequency_range_equal_bounds_is_a_no_op():
    reference = _crop_context(None).prepared_data()
    same = _crop_context(
        {"minimum_frequency": 20.0, "maximum_frequency": 1024.0}
    ).prepared_data()
    assert torch.equal(reference, same)


def test_frequency_range_narrowing_requires_crop_license():
    with pytest.raises(ValueError, match="Cropping disabled"):
        _crop_context({"minimum_frequency": 25.0}).prepared_data()


def test_frequency_range_beyond_domain_rejected():
    # The O1-file-reuse scenario: event bounds wider than the network domain.
    with pytest.raises(ValueError, match="domain.f_max"):
        _crop_context({"maximum_frequency": 1099.0}).prepared_data()


def test_context_without_event_data_serves_metadata_views_only():
    # A stripped payload (settings but no strain data) still yields the
    # metadata-derived views; only the likelihood demands event data.
    ctx = GWSamplerContext.from_model_metadata(_MODEL_METADATA, event_data=None)
    assert ctx.detectors == ["H1", "L1"]
    assert ctx.domain.f_max == _DOMAIN_SETTINGS["f_max"]
    with pytest.raises(ValueError, match="event data"):
        ctx.likelihood()


def test_sampler_provenance_block():
    import ast

    from dingo.core.factors import ChainComposer, DeltaFactor
    from dingo.gw.inference.sampler import GWComposedSampler

    sampler = GWComposedSampler(
        ChainComposer([DeltaFactor({"a": 1.0})]), None, {}, ["a"]
    )
    sampler.provenance_extra["models"] = {"model": "model.pt"}
    block = sampler.sampler_provenance()
    assert block["version"] == 1 and block["implementation"] == "composed"
    assert block["chain"][0]["step"] == "DeltaFactor"
    assert block["models"] == {"model": "model.pt"}
    assert ast.literal_eval(str(block)) == block


def test_frequency_range_cropping_masks_network_input():
    ctx = _crop_context(
        {"minimum_frequency": 25.0},
        crop_settings={"cropping_probability": 0.5, "f_min_upper": 30.0},
    )
    out = ctx.prepared_data()  # (n_det, 3 channels, n_bins), real strain first
    frequencies = ctx.domain.sample_frequencies[ctx.domain.min_idx :]
    strain_real = out[0, 0].numpy()
    assert (strain_real[frequencies < 25.0] == 0).all()
    assert (strain_real[frequencies >= 25.0] != 0).all()


def test_likelihood_caller_marginalization_bounds_win(context):
    # Bounds provided by the caller (e.g. from an updated prior at the IS layer)
    # are used as-is; the network-prior fill only runs when they are missing.
    likelihood = context.likelihood(
        time_marginalization_kwargs={"n_fft": 2, "t_lower": 1.0, "t_upper": 2.0}
    )
    kwargs = likelihood.kwargs["time_marginalization_kwargs"]
    assert (kwargs["t_lower"], kwargs["t_upper"]) == (1.0, 2.0)


def test_context_device_default_and_explicit():
    ctx = GWSamplerContext(
        domain=None,
        data_prep=None,
        event_data={},
        device="meta",
    )
    assert ctx.device == "meta"
    ctx_default = GWSamplerContext(domain=None, data_prep=None, event_data={})
    assert ctx_default.device == "cpu"


# Full conditional-model metadata, as serialized in Result.settings.
_MODEL_METADATA = {
    "dataset_settings": {
        "domain": _DOMAIN_SETTINGS,
        "waveform_generator": {"approximant": "IMRPhenomD", "f_ref": 20.0},
    },
    "train_settings": {
        "data": {
            "detectors": ["H1", "L1"],
            "ref_time": 1126259462.4,
            "inference_parameters": ["chirp_mass"],
        }
    },
}


class _StubModel:
    metadata = _MODEL_METADATA
    device = "cpu"


def test_from_model_metadata_matches_from_model():
    ctx_meta = GWSamplerContext.from_model_metadata(_MODEL_METADATA, event_data={})
    ctx_model = GWSamplerContext.from_model(_StubModel(), event_data={})
    for ctx in (ctx_meta, ctx_model):
        assert ctx.detectors == ["H1", "L1"]
        assert ctx.t_ref == 1126259462.4
        assert ctx.domain.domain_dict == ctx_meta.domain.domain_dict
        assert ctx.model_metadata is _MODEL_METADATA
        assert ctx.device == "cpu"
        assert ctx._data_prep is not None


def test_from_model_rejects_unconditional():
    nde_metadata = copy.deepcopy(_MODEL_METADATA)
    nde_metadata["train_settings"]["data"]["unconditional"] = True
    nde_metadata["base"] = _MODEL_METADATA

    class _StubNDE:
        metadata = nde_metadata
        device = "cpu"

    with pytest.raises(ValueError, match="unconditional"):
        GWSamplerContext.from_model(_StubNDE(), event_data={})
    # The analysis views remain available from the base metadata.
    ctx = GWSamplerContext.from_model_metadata(nde_metadata["base"], event_data={})
    assert ctx.detectors == ["H1", "L1"]
