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


def test_likelihood_cache_handles_domain_and_none_settings(context):
    # A bare call (data_domain=None) followed by an explicit domain (either order)
    # must be a clean cache miss, not a Domain == None comparison error.
    bare = context.likelihood()
    explicit = context.likelihood(data_domain=build_domain(_DOMAIN_SETTINGS))
    assert explicit is not bare
    assert context.likelihood() is not explicit


def test_domain_eq_is_none_safe():
    domain = build_domain(_DOMAIN_SETTINGS)
    assert domain == build_domain(_DOMAIN_SETTINGS)
    assert domain != None  # noqa: E711
    assert (domain == 3) is False


def test_likelihood_importance_sampling_overrides(context):
    # IS-state overrides pass through: an explicit data domain, an updated
    # waveform-generator delta_f, and an explicit frequency range.
    likelihood = context.likelihood(
        data_domain="rebuilt-domain",
        wfg_delta_f=0.5,
        frequency_update={"minimum_frequency": 21.0, "maximum_frequency": 512.0},
    )
    assert likelihood.kwargs["data_domain"] == "rebuilt-domain"
    assert likelihood.kwargs["wfg_domain"].delta_f == 0.5
    assert likelihood.kwargs["frequency_update"]["maximum_frequency"] == 512.0
    # Defaults reproduce the context's own views.
    default = context.likelihood()
    assert default.kwargs["data_domain"] is context.domain
    assert default.kwargs["wfg_domain"].delta_f == _DOMAIN_SETTINGS["delta_f"]


def test_prepared_data_rejects_frequency_cropping():
    # A narrower event frequency range needs input masking, which the composed
    # data-prep chain does not implement yet -> loud failure. Bounds equal to or
    # wider than the domain (generation writes base-domain bounds) are fine.
    def make(event_metadata):
        return GWSamplerContext(
            domain=build_domain(_DOMAIN_SETTINGS),
            detectors=["H1"],
            t_ref=0.0,
            data_prep=lambda data: "prepared",
            event_data={},
            event_metadata=event_metadata,
        )

    with pytest.raises(NotImplementedError, match="cropping"):
        make({"minimum_frequency": 25.0}).prepared_data()
    with pytest.raises(NotImplementedError, match="cropping"):
        make({"maximum_frequency": {"H1": 512.0}}).prepared_data()
    assert make({"minimum_frequency": 20.0}).prepared_data() == "prepared"
    assert make({"maximum_frequency": 1099.0}).prepared_data() == "prepared"
    assert make(None).prepared_data() == "prepared"


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
