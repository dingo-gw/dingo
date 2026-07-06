"""
Unit tests for ``GWSamplerContext`` plumbing: the default-likelihood cache, the
device attribute, and the metadata constructors. The likelihood itself is stubbed
(no waveform generation); the model-based construction parity lives in the local
harnesses.
"""

import copy

import pytest

import dingo.gw.inference.factors as factors_module
from dingo.gw.domains import build_domain
from dingo.gw.inference.factors import GWSamplerContext

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
    "train_settings": {"data": {"inference_parameters": ["chirp_mass"]}},
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
        factors_module, "StationaryGaussianGWLikelihood", _StubLikelihood
    )
    _StubLikelihood.constructions = 0
    return GWSamplerContext(
        domain=build_domain(_DOMAIN_SETTINGS),
        detectors=["H1"],
        t_ref=1126259462.4,
        data_prep=None,
        event_data={},
        base_metadata=_BASE_METADATA,
    )


def test_likelihood_rebuilt_only_when_settings_change(context):
    # Repeated calls with unchanged arguments share the last-built instance (the
    # synthetic-phase factor requests one per chain chunk); a settings change
    # builds a replacement.
    default = context.likelihood()
    assert context.likelihood() is default
    assert _StubLikelihood.constructions == 1

    base = context.likelihood(use_base_domain=True)
    assert base is not default
    assert base.kwargs["use_base_domain"] is True
    assert context.likelihood(use_base_domain=True) is base
    assert _StubLikelihood.constructions == 2


def test_context_device_default_and_explicit():
    ctx = GWSamplerContext(
        domain=None,
        detectors=["H1"],
        t_ref=0.0,
        data_prep=None,
        event_data={},
        device="meta",
    )
    assert ctx.device == "meta"
    ctx_default = GWSamplerContext(
        domain=None, detectors=["H1"], t_ref=0.0, data_prep=None, event_data={}
    )
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
        assert ctx.base_metadata is _MODEL_METADATA
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
