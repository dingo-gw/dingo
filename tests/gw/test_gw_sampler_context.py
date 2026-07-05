"""
Unit tests for ``GWSamplerContext`` plumbing: the default-likelihood cache and the
device attribute. The likelihood itself is stubbed (no waveform generation); the
model-based construction parity lives in the local harnesses.
"""

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
