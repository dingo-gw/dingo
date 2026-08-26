"""
Chain shapes assembled by the GWComposedSampler builders on stub model metadata (no
network), and the builders' rejections of inconsistent inputs.
"""

import copy

import numpy as np
import pytest

from dingo.core.inference.steps import DeltaFactor
from dingo.gw.inference.sampler import (
    GWComposedSampler,
    _assert_consistent_gnpe_data_prep,
)

DETECTORS = ["H1", "L1"]
INFERENCE_PARAMETERS = ["chirp_mass", "mass_ratio", "ra", "dec", "geocent_time"]
CONTEXT_PARAMETERS = ["L1_time_proxy_relative"]


def _metadata(inference_parameters, context_parameters, gnpe, ref_time=1126259462.4):
    standardized = inference_parameters + context_parameters
    standardized += ["H1_time", "L1_time", "H1_time_proxy", "L1_time_proxy"]
    metadata = {
        "dataset_settings": {
            "domain": {
                "type": "UniformFrequencyDomain",
                "f_min": 20.0,
                "f_max": 256.0,
                "delta_f": 0.5,
            },
            "waveform_generator": {"approximant": "IMRPhenomD", "f_ref": 20.0},
            "intrinsic_prior": {
                "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25, maximum=31)",
                "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1)",
                "a_1": 0.0,
                "a_2": 0.0,
                "phase": "default",
            },
        },
        "train_settings": {
            "data": {
                "detectors": DETECTORS,
                "ref_time": ref_time,
                "extrinsic_prior": {
                    "dec": "default",
                    "ra": "default",
                    "geocent_time": "default",
                    "luminosity_distance": "default",
                    "psi": "default",
                },
                "inference_parameters": inference_parameters,
                "context_parameters": context_parameters,
                "standardization": {
                    "mean": {p: 0.0 for p in standardized},
                    "std": {p: 1.0 for p in standardized},
                },
            }
        },
    }
    if gnpe:
        metadata["train_settings"]["data"]["gnpe_time_shifts"] = {
            "kernel": "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)",
            "exact_equiv": True,
        }
    return metadata


class _StubModel:
    device = "cpu"

    def __init__(self, metadata):
        self.metadata = metadata
        self.base_metadata = metadata


MAIN = _StubModel(_metadata(INFERENCE_PARAMETERS, CONTEXT_PARAMETERS, gnpe=True))
INIT = _StubModel(_metadata(["H1_time", "L1_time"], [], gnpe=False))
PLAIN = _StubModel(_metadata(INFERENCE_PARAMETERS, [], gnpe=False))
EVENT_DATA = {
    "waveform": {d: np.ones(513, dtype=complex) for d in DETECTORS},
    "asds": {d: np.ones(513) for d in DETECTORS},
}


def _step_names(sampler):
    return [type(step).__name__ for step in sampler.composer.steps]


def test_from_gnpe_models_chain_shape():
    sampler = GWComposedSampler.from_gnpe_models(
        INIT, MAIN, EVENT_DATA, num_iterations=3
    )
    assert _step_names(sampler) == ["GibbsBlock", "RAToEventFrame", "DeltaFactor"]
    gibbs = sampler.composer.steps[0]
    assert gibbs.num_iterations == 3
    # The block emits the proxies and the (aliased) inference parameters.
    assert set(gibbs.parameters) >= {"H1_time_proxy", "L1_time_proxy", "ra@t_ref"}


def test_from_singlestep_gnpe_chain_shape():
    pins = DeltaFactor({"H1_time_proxy": 0.0, "L1_time_proxy": 0.0})
    sampler = GWComposedSampler.from_singlestep_gnpe(MAIN, pins, EVENT_DATA)
    assert _step_names(sampler) == [
        "DeltaFactor",
        "GNPEFlowFactor",
        "GNPEKernelCorrection",
        "RAToEventFrame",
        "DeltaFactor",
    ]
    flow, correction = sampler.composer.steps[1], sampler.composer.steps[2]
    assert flow.conditioning == ["H1_time_proxy", "L1_time_proxy"]
    assert sorted(correction.consumes) == ["H1_time", "L1_time"]


def test_gnpe_data_prep_mismatch_is_rejected():
    bad_init = _StubModel(
        _metadata(["H1_time", "L1_time"], [], gnpe=False, ref_time=0.0)
    )
    with pytest.raises(ValueError):
        _assert_consistent_gnpe_data_prep(bad_init, MAIN)


def test_from_model_rejects_a_time_gnpe_model():
    # A time-GNPE main model's data must be time-shifted by the proxies, which the
    # plain single-network chain does not do.
    with pytest.raises(ValueError, match="time-GNPE"):
        GWComposedSampler.from_model(
            MAIN, EVENT_DATA, fixed_context_parameters={"L1_time_proxy_relative": 0.0}
        )


def test_from_model_rejects_pins_the_model_does_not_condition_on():
    with pytest.raises(ValueError, match="exactly these keys"):
        GWComposedSampler.from_model(
            PLAIN, EVENT_DATA, fixed_context_parameters={"chirp_mass_proxy": 1.2}
        )


def test_from_model_plain_chain_shape():
    sampler = GWComposedSampler.from_model(PLAIN, EVENT_DATA)
    assert _step_names(sampler) == ["FlowFactor", "RAToEventFrame", "DeltaFactor"]
    assert sampler.metadata is PLAIN.base_metadata
