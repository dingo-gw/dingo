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


def test_heterodyne_inserted_before_decimation_draws_from_conditioning():
    ctx = GWSamplerContext.from_model_metadata(_BNS_METADATA, _event_data())
    first = ctx._data_prep.transforms[0]
    assert isinstance(first, HeterodynePhase)
    # Parameters mode: the transform draws the chirp mass from the sample dict;
    # no value is baked into the context. The context records only the names its
    # preparation is a function of.
    assert first.fixed_parameters is None
    assert first.order == 0 and first.inverse is False
    assert ctx.data_prep_conditioning == ["chirp_mass_proxy"]


def test_prepared_data_requires_conditioning():
    # Without conditioning the context still constructs (the likelihood and
    # prior views must work for from-file results), but the network-input view
    # is a function of the chain conditioning and fails loudly without it.
    ctx = GWSamplerContext.from_model_metadata(_BNS_METADATA, _event_data())
    assert ctx.prior is not None
    with pytest.raises(ValueError, match="chirp_mass_proxy"):
        ctx.prepared_data()


def _conditioning(value=1.1975, n=4):
    import torch

    return {
        "chirp_mass_proxy": torch.full((n,), value),
        "ra": torch.full((n,), _PINS["ra"]),
        "dec": torch.full((n,), _PINS["dec"]),
    }


def test_prepared_data_pinned_is_row_aligned_and_memoized():
    import torch

    ctx = GWSamplerContext.from_model_metadata(_BNS_METADATA, _event_data())
    prepared = ctx.prepared_data(conditioning=_conditioning())
    # Row-aligned: one data row per conditioning row, the single pinned
    # preparation viewed across them.
    assert prepared.shape[0] == 4
    cached = ctx._prepared
    assert torch.equal(ctx.prepared_data(conditioning=_conditioning()), prepared)
    assert ctx._prepared is cached
    # A different pin recomputes correctly (the memo is keyed on the value).
    other = ctx.prepared_data(conditioning=_conditioning(value=1.30))
    assert ctx._prepared is not cached
    assert not torch.equal(other, prepared)


def test_prepared_data_varying_matches_single_preparations():
    import torch

    values = [1.19, 1.20, 1.21, 1.22]
    ctx = GWSamplerContext.from_model_metadata(_BNS_METADATA, _event_data())
    conditioning = _conditioning()
    conditioning["chirp_mass_proxy"] = torch.tensor(values, dtype=torch.float64)
    batch = ctx.prepared_data(conditioning=conditioning)
    assert batch.shape[0] == len(values)
    # Row i equals the pinned preparation at value i (compared at equal pin
    # precision: the pinned path extracts the float64 value).
    for i, value in enumerate(values):
        single_conditioning = _conditioning()
        single_conditioning["chirp_mass_proxy"] = torch.full(
            (4,), value, dtype=torch.float64
        )
        single = GWSamplerContext.from_model_metadata(
            _BNS_METADATA, _event_data()
        ).prepared_data(conditioning=single_conditioning)
        assert torch.equal(batch[i], single[0])
    # A varying pass is not memoized, and the pinned path still works after.
    assert ctx._prepared is None
    assert ctx.prepared_data(conditioning=_conditioning()).shape[0] == 4


def test_prepared_data_varying_rows_follow_values():
    import torch

    ctx = GWSamplerContext.from_model_metadata(_BNS_METADATA, _event_data())
    conditioning = _conditioning()
    conditioning["chirp_mass_proxy"] = torch.tensor(
        [1.19, 1.19, 1.21, 1.21], dtype=torch.float64
    )
    batch = ctx.prepared_data(conditioning=conditioning)
    assert torch.equal(batch[0], batch[1])
    assert torch.equal(batch[2], batch[3])
    assert not torch.equal(batch[0], batch[2])


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


def test_chirp_mass_scan_grid_spans_prior_at_kernel_spacing():
    from dingo.gw.inference.scan import chirp_mass_scan_grid

    grid = chirp_mass_scan_grid(_BNS_METADATA, overlap_factor=2)
    # Prior [1.0, 2.0], kernel width 0.01, overlap 2 -> 200 points inset by the
    # kernel edges, spaced at most half a kernel width apart.
    assert len(grid) == 200
    assert grid[0] == pytest.approx(1.005)
    assert grid[-1] == pytest.approx(1.995)
    assert np.diff(grid).max() <= 0.005 + 1e-12


def test_chirp_mass_scan_validates_model_and_pins():
    import copy

    from dingo.gw.inference.scan import chirp_mass_scan

    # The non-proxy context parameters must be pinned by the caller.
    with pytest.raises(ValueError, match="remaining context parameters"):
        chirp_mass_scan(_StubBNSModel(), _event_data(), fixed_context_parameters={})

    # A model without chirp-mass conditioning cannot be scanned.
    metadata_no_chirp = copy.deepcopy(_BNS_METADATA)
    del metadata_no_chirp["train_settings"]["data"]["gnpe_chirp"]
    metadata_no_chirp["train_settings"]["data"]["context_parameters"] = ["ra", "dec"]

    class _StubNonChirpModel:
        metadata = metadata_no_chirp
        device = "cpu"

    with pytest.raises(ValueError, match="chirp_mass_proxy"):
        chirp_mass_scan(_StubNonChirpModel(), _event_data())
