"""
Unit tests for ``GWSamplerContext`` plumbing: the likelihood cache and its
base-domain keyword, the grid the likelihood takes from the event record, the
device attribute, and the metadata constructors. The likelihood itself is stubbed
(no waveform generation); the model-based construction parity lives in the local
harnesses.
"""

import copy

import numpy as np
import pytest
import torch

import dingo.gw.inference.context as context_module
from dingo.gw.domains import UniformFrequencyDomain, build_domain
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
        event_data=_event_data(_bins(_DOMAIN_SETTINGS["f_max"])),
        model_metadata=_BASE_METADATA,
    )


def _bins(f_max, delta_f=_DOMAIN_SETTINGS["delta_f"]):
    return int(f_max / delta_f) + 1


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


def test_likelihood_cache_keyed_on_base_domain_choice(context):
    # The base-domain choice is a likelihood argument like the marginalizations,
    # keyed into the cache; nothing about the context changes.
    default = context.likelihood()
    base = context.likelihood(use_base_domain=True)
    assert base is not default
    assert base.kwargs["use_base_domain"] is True
    assert context.likelihood(use_base_domain=True) is base
    assert _StubLikelihood.constructions == 2


def _grid(f_min, f_max, delta_f):
    return {
        "type": "UniformFrequencyDomain",
        "f_min": f_min,
        "f_max": f_max,
        "delta_f": delta_f,
    }


def test_likelihood_grid_comes_from_the_event_record(monkeypatch):
    # The pipe records the grid it generated the data on; the likelihood uses
    # that grid, and the waveform generator extends to contain it. Without a
    # record (older files) the data are on the network's grid.
    monkeypatch.setattr(
        context_module, "StationaryGaussianGWLikelihood", _StubLikelihood
    )
    delta_f = _DOMAIN_SETTINGS["delta_f"]
    for event_metadata in (None, {"domain": _grid(20.0, 1024.0, delta_f)}):
        same = GWSamplerContext.from_model_metadata(
            _BASE_METADATA, _event_data(_bins(1024.0)), event_metadata
        )
        kwargs = same.likelihood().kwargs
        assert kwargs["data_domain"] is same.domain
        assert kwargs["wfg_domain"] == same.domain
    # Data generated for a wider range, from a lower bound, for importance sampling.
    wide = GWSamplerContext.from_model_metadata(
        _BASE_METADATA,
        _event_data(_bins(2048.0)),
        event_metadata={
            "domain": _grid(15.0, 2048.0, delta_f),
            "maximum_frequency": 2048.0,
        },
    )
    kwargs = wide.likelihood().kwargs
    assert wide.domain.f_max == 1024.0  # the network's, unchanged
    assert kwargs["data_domain"] == build_domain(_grid(15.0, 2048.0, delta_f))
    assert (kwargs["wfg_domain"].f_min, kwargs["wfg_domain"].f_max) == (15.0, 2048.0)
    assert kwargs["frequency_update"]["maximum_frequency"] == 2048.0
    # Data generated at another duration.
    longer = GWSamplerContext.from_model_metadata(
        _BASE_METADATA,
        _event_data(_bins(1024.0, 0.125)),
        event_metadata={"domain": _grid(20.0, 1024.0, 0.125), "T": 8.0},
    )
    kwargs = longer.likelihood().kwargs
    assert kwargs["data_domain"].delta_f == 0.125
    assert kwargs["wfg_domain"].delta_f == 0.125


def test_multibanded_decimated_likelihood_needs_the_network_grid(monkeypatch):
    # A multibanded model decimates data on its base grid; data generated for
    # another range must be evaluated on the base domain.
    monkeypatch.setattr(
        context_module, "StationaryGaussianGWLikelihood", _StubLikelihood
    )
    metadata = copy.deepcopy(_BASE_METADATA)
    metadata["dataset_settings"]["domain"] = {
        "type": "MultibandedFrequencyDomain",
        "nodes": [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        "delta_f_initial": 0.0625,
        "base_domain": _grid(20.0, 2048.0, 0.0625),
    }
    on_grid = GWSamplerContext.from_model_metadata(
        metadata,
        _event_data(_bins(2048.0, 0.0625)),
        event_metadata={"domain": _grid(20.0, 2048.0, 0.0625)},
    )
    assert on_grid.likelihood().kwargs["data_domain"] is on_grid.domain
    # On the network's grid the likelihood class does the base-domain switch.
    base = on_grid.likelihood(use_base_domain=True).kwargs
    assert base["data_domain"] is on_grid.domain and base["use_base_domain"] is True
    off_grid = GWSamplerContext.from_model_metadata(
        metadata,
        _event_data(_bins(4096.0, 0.0625)),
        event_metadata={"domain": _grid(20.0, 4096.0, 0.0625)},
    )
    with pytest.raises(ValueError, match="base domain"):
        off_grid.likelihood()
    assert off_grid.likelihood(use_base_domain=True).kwargs["data_domain"].f_max == (
        4096.0
    )


def test_prepared_data_refuses_data_off_the_network_grid():
    wide = GWSamplerContext.from_model_metadata(
        _MODEL_METADATA,
        _event_data(_bins(2048.0)),
        event_metadata={"domain": _grid(20.0, 2048.0, _DOMAIN_SETTINGS["delta_f"])},
    )
    with pytest.raises(ValueError, match="network's grid"):
        wide.prepared_data()


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
        event_data=_event_data(_bins(_DOMAIN_SETTINGS["f_max"])),
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

    from dingo.core.inference.composer import ChainComposer
    from dingo.core.inference.steps import DeltaFactor
    from dingo.gw.inference.sampler import GWComposedSampler

    sampler = GWComposedSampler(ChainComposer([DeltaFactor({"a": 1.0})]), None)
    sampler.provenance_extra["models"] = {"model": "model.pt"}
    block = sampler.sampler_provenance()
    assert set(block) == {"chain", "models"}
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
    # are used as-is; the context requires them and fills nothing itself.
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
