"""Unit tests for the fixed-proxy (DINGO-BNS) chain construction: the heterodyne
insertion into the context's data preparation, the pin requirement, and the
chain assembly `[DeltaFactor(pins), conditioned FlowFactor, ProxyOffsetReparam]`.
Mock metadata only; the real-model smoke lives in the local harnesses."""

import numpy as np
import pytest

from dingo.core.factors import DeltaFactor, FlowFactor, ProxyOffsetReparam
from dingo.gw.inference.factors import (
    GWComposedSampler,
    GWSamplerContext,
    _proxy_offset_steps,
)
from dingo.gw.transforms import HeterodynePhase

_PINS = {"chirp_mass_proxy": 1.1975, "ra": 3.44616, "dec": -0.408084}

_BNS_METADATA = {
    "dataset_settings": {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 256.0,
            "delta_f": 0.25,
        },
        "waveform_generator": {"approximant": "IMRPhenomPv2_NRTidal", "f_ref": 100.0},
        "intrinsic_prior": {
            "chirp_mass": (
                "bilby.gw.prior.UniformInComponentsChirpMass(minimum=1.0, maximum=2.0)"
            ),
            "mass_ratio": (
                "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.5, maximum=1.0)"
            ),
        },
    },
    "train_settings": {
        "data": {
            "detectors": ["H1"],
            "ref_time": 1187008882.42,
            "extrinsic_prior": {"ra": "default", "dec": "default"},
            "inference_parameters": ["delta_chirp_mass", "mass_ratio"],
            "context_parameters": ["ra", "dec", "chirp_mass_proxy"],
            "gnpe_chirp": {
                "kernel": {
                    "chirp_mass": (
                        "bilby.core.prior.Uniform(minimum=-0.005, maximum=0.005)"
                    )
                },
                "order": 0,
            },
            "standardization": {
                "mean": {
                    "delta_chirp_mass": 0.0,
                    "mass_ratio": 0.75,
                    "ra": np.pi,
                    "dec": 0.0,
                    "chirp_mass_proxy": 1.5,
                },
                "std": {
                    "delta_chirp_mass": 0.003,
                    "mass_ratio": 0.15,
                    "ra": 1.8,
                    "dec": 0.6,
                    "chirp_mass_proxy": 0.3,
                },
            },
        }
    },
}


class _StubBNSModel:
    metadata = _BNS_METADATA
    device = "cpu"


def _event_data(n_bins=1025):
    strain = np.ones(n_bins, dtype=complex)
    return {"waveform": {"H1": strain}, "asds": {"H1": np.ones(n_bins)}}


def test_heterodyne_inserted_before_decimation_with_fixed_proxy():
    ctx = GWSamplerContext.from_model_metadata(
        _BNS_METADATA, _event_data(), fixed_context_parameters=_PINS
    )
    first = ctx._data_prep.transforms[0]
    assert isinstance(first, HeterodynePhase)
    # Only proxy entries parameterize the data preparation; the pinned sky
    # position is conditioning only.
    assert first.fixed_parameters == {"chirp_mass": _PINS["chirp_mass_proxy"]}
    assert first.order == 0 and first.inverse is False


def test_missing_proxy_makes_prepared_data_fail_loudly():
    # Without pins the context still constructs (the likelihood and prior views
    # must work for from-file results), but the network-input view is undefined.
    ctx = GWSamplerContext.from_model_metadata(_BNS_METADATA, _event_data())
    assert ctx.prior is not None
    with pytest.raises(ValueError, match="chirp-mass"):
        ctx.prepared_data()


def test_proxy_offset_step_selection():
    steps = _proxy_offset_steps(
        ["delta_chirp_mass", "mass_ratio"], ["ra", "dec", "chirp_mass_proxy"]
    )
    assert len(steps) == 1 and steps[0].parameters == ["chirp_mass"]
    # No offset step without a matching pinned proxy.
    assert _proxy_offset_steps(["delta_chirp_mass"], ["ra"]) == []


def test_from_model_assembles_pinned_chain():
    sampler = GWComposedSampler.from_model(
        _StubBNSModel(), _event_data(), fixed_context_parameters=_PINS
    )
    kinds = [type(s).__name__ for s in sampler.composer.steps]
    assert kinds == ["DeltaFactor", "FlowFactor", "ProxyOffsetReparam"]
    delta, flow, offset = sampler.composer.steps
    assert set(delta.parameters) == set(_PINS)
    assert flow.conditioning == ["ra", "dec", "chirp_mass_proxy"]
    assert offset.consumes == ["delta_chirp_mass"]


def test_from_model_requires_all_pins():
    with pytest.raises(ValueError, match="conditions on"):
        GWComposedSampler.from_model(_StubBNSModel(), _event_data())
    with pytest.raises(ValueError, match="conditions on"):
        GWComposedSampler.from_model(
            _StubBNSModel(),
            _event_data(),
            fixed_context_parameters={"chirp_mass_proxy": 1.2},
        )
