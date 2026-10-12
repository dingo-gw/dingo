"""
Time GNPE with distance prior conditioning, on a mock main network: the pinned bound
reaches the network (standardized, after the relative proxy) on both the sampling and
the log_prob paths, the single-step chain re-plugs, the Gibbs block carries the pin
through every sweep, and the builders check the pins.
"""

import copy
import math

import numpy as np
import pytest
import torch
from astropy.utils import iers

from dingo.core.inference.composer import ChainComposer, GibbsBlock
from dingo.core.inference.steps import DeltaFactor, Factor
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.sampler import GWComposedSampler
from dingo.gw.inference.steps import (
    GNPEFlowFactor,
    GNPEKernelCorrection,
    GNPEKernelFactor,
    RAToEventFrame,
)

iers.conf.auto_download = False

INFERENCE_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "geocent_time",
    "luminosity_distance",
    "ra",
    "dec",
]
CONTEXT_PARAMETERS = ["L1_time_proxy_relative", "log_luminosity_distance_max"]
REF_TIME = 1126259462.4
D_MAX = 3000.0
LOG_D_MAX_MEAN, LOG_D_MAX_STD = 7.0, 1.5
METADATA = {
    "dataset_settings": {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 64.0,
            "delta_f": 0.25,
        },
        "waveform_generator": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        "intrinsic_prior": {
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25, maximum=31)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1)",
        },
    },
    "train_settings": {
        "data": {
            "detectors": ["H1", "L1"],
            "ref_time": REF_TIME,
            "extrinsic_prior": {
                "ra": "default",
                "dec": "default",
                "geocent_time": "default",
                "luminosity_distance": "bilby.gw.prior.UniformComovingVolume("
                "minimum=100.0, maximum=8000.0, name='luminosity_distance')",
            },
            "inference_parameters": INFERENCE_PARAMETERS,
            "context_parameters": CONTEXT_PARAMETERS,
            "gnpe_time_shifts": {
                "kernel": "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)",
                "exact_equiv": True,
            },
            "distance_prior_conditioning": {
                "luminosity_distance_max": "bilby.core.prior.LogUniform("
                "minimum=500.0, maximum=8000.0)"
            },
            "standardization": {
                "mean": {
                    "chirp_mass": 28.0,
                    "mass_ratio": 0.6,
                    "geocent_time": 0.0,
                    "luminosity_distance": 2000.0,
                    "ra": 3.1,
                    "dec": 0.0,
                    "L1_time_proxy_relative": 0.0,
                    "log_luminosity_distance_max": LOG_D_MAX_MEAN,
                },
                "std": {
                    "chirp_mass": 2.0,
                    "mass_ratio": 0.2,
                    "geocent_time": 0.01,
                    "luminosity_distance": 1500.0,
                    "ra": 0.3,
                    "dec": 0.3,
                    "L1_time_proxy_relative": 0.003,
                    "log_luminosity_distance_max": LOG_D_MAX_STD,
                },
            },
        }
    },
}
PINS = {"log_luminosity_distance_max": math.log(D_MAX)}
PIN_STANDARDIZED = (math.log(D_MAX) - LOG_D_MAX_MEAN) / LOG_D_MAX_STD


class _MockGNPEModel:
    """A GNPE main network with an analytic density that depends on the context
    (standard normal in the standardized parameters, plus a context-dependent offset),
    recording the context it is given."""

    device = "cpu"
    network = torch.nn.Identity()

    def __init__(self, metadata=METADATA):
        self.metadata = metadata
        self.base_metadata = metadata
        self.contexts = []

    @staticmethod
    def _log_prob(z, ctx):
        weights = torch.arange(1, ctx.shape[-1] + 1, dtype=ctx.dtype)
        return (
            -0.5 * (z**2).sum(-1)
            - 0.5 * z.shape[-1] * math.log(2 * math.pi)
            + 0.3 * (ctx * weights).sum(-1)
        )

    def sample_and_log_prob(self, data, ctx, num_samples=1):
        self.contexts.append(ctx.clone())
        n_par = len(self.metadata["train_settings"]["data"]["inference_parameters"])
        z = torch.randn(data.shape[0], num_samples, n_par)
        return z, self._log_prob(z, ctx.unsqueeze(1))

    def log_prob(self, z, data, ctx):
        self.contexts.append(ctx.clone())
        return self._log_prob(z, ctx)


class _DetectorTimesInit(Factor):
    """A stand-in for the init network: Gaussian detector times."""

    parameters = ["H1_time", "L1_time"]
    conditioning = []

    def sample_and_log_prob(self, num_samples, context, given=None):
        z = 0.005 * torch.randn(num_samples, 2)
        return {"H1_time": z[:, 0], "L1_time": z[:, 1]}, torch.zeros(num_samples)

    def log_prob(self, theta_i, context, given=None):
        return torch.zeros_like(theta_i["H1_time"])


@pytest.fixture
def context():
    n_bins = int(64 / 0.25) + 1
    event_data = {
        "waveform": {d: np.ones(n_bins, dtype=complex) for d in ["H1", "L1"]},
        "asds": {d: np.ones(n_bins) for d in ["H1", "L1"]},
    }
    return GWSamplerContext.from_model_metadata(
        METADATA, event_data, event_metadata={"time_event": REF_TIME + 7200}
    )


def _event_data():
    n_bins = int(64 / 0.25) + 1
    return {
        "waveform": {d: np.ones(n_bins, dtype=complex) for d in ["H1", "L1"]},
        "asds": {d: np.ones(n_bins) for d in ["H1", "L1"]},
    }


def _proxies_and_pins(kernel, context, n, log_d_max=math.log(D_MAX)):
    times = {"H1_time": 0.01 * torch.randn(n), "L1_time": 0.01 * torch.randn(n)}
    proxies, _ = kernel.sample_and_log_prob(1, context, times)
    return {**proxies, "log_luminosity_distance_max": torch.full((n,), log_d_max)}


def test_flow_factor_conditions_on_proxies_and_the_pin():
    flow = GNPEFlowFactor(_MockGNPEModel(), aliases={"ra": "ra@t_ref"})
    assert flow.pinned_parameters == ["log_luminosity_distance_max"]
    assert flow.conditioning == [
        "H1_time_proxy",
        "L1_time_proxy",
        "log_luminosity_distance_max",
    ]


def test_pin_reaches_the_network_standardized_on_both_paths(context):
    torch.manual_seed(0)
    model = _MockGNPEModel()
    flow = GNPEFlowFactor(model, aliases={"ra": "ra@t_ref"})
    given = _proxies_and_pins(GNPEKernelFactor(model), context, 40)
    out, lp = flow.sample_and_log_prob(1, context, given)
    theta_i = {p: out[p] for p in flow.parameters}
    lp_replug = flow.log_prob(theta_i, context, given)
    assert torch.allclose(lp_replug, lp, atol=1e-5)
    for ctx in model.contexts:
        assert ctx.shape == (40, 2)
        # Column order follows context_parameters: relative proxy, then the pin.
        assert torch.allclose(ctx[:, 1], torch.full((40,), PIN_STANDARDIZED))
        expected_relative = (
            given["L1_time_proxy"] - given["H1_time_proxy"]
        ) / 0.003
        assert torch.allclose(ctx[:, 0], expected_relative, atol=1e-4)


def test_pin_changes_the_density(context):
    # The network density must depend on the pin (else it is silently dropped).
    torch.manual_seed(1)
    model = _MockGNPEModel()
    flow = GNPEFlowFactor(model, aliases={"ra": "ra@t_ref"})
    given = _proxies_and_pins(GNPEKernelFactor(model), context, 10)
    out, _ = flow.sample_and_log_prob(1, context, given)
    theta_i = {p: out[p] for p in flow.parameters}
    other = {**given, "log_luminosity_distance_max": torch.full((10,), math.log(800.0))}
    shift = flow.log_prob(theta_i, context, other) - flow.log_prob(
        theta_i, context, given
    )
    expected = 0.3 * 2 * (math.log(800.0) - math.log(D_MAX)) / LOG_D_MAX_STD
    assert torch.allclose(shift, torch.full((10,), expected), atol=1e-4)


def test_unsupported_pin_is_refused():
    metadata = copy.deepcopy(METADATA)
    data = metadata["train_settings"]["data"]
    data["context_parameters"] = ["L1_time_proxy_relative", "chirp_mass_proxy"]
    with pytest.raises(NotImplementedError, match="chirp_mass_proxy"):
        GNPEFlowFactor(_MockGNPEModel(metadata))


@pytest.mark.parametrize("num_samples", [1, 5])
def test_singlestep_chain_with_pins_replugs(context, num_samples):
    torch.manual_seed(2)
    model = _MockGNPEModel()
    flow = GNPEFlowFactor(model, aliases={"ra": "ra@t_ref"})
    kernel = GNPEKernelFactor(model)
    chain = ChainComposer(
        [
            DeltaFactor(PINS),
            DeltaFactor({"H1_time_proxy": 0.0, "L1_time_proxy": 0.001}),
            flow,
            GNPEKernelCorrection(kernel),
            RAToEventFrame(),
        ]
    )
    samples, lp = chain.sample_and_log_prob(num_samples, context)
    assert lp.shape == (num_samples,)
    assert torch.allclose(
        samples["log_luminosity_distance_max"],
        torch.full((num_samples,), math.log(D_MAX)),
    )
    assert torch.allclose(chain.log_prob(samples, context), lp, atol=1e-5)


def test_gibbs_block_carries_the_pin_through_every_sweep(context):
    torch.manual_seed(3)
    model = _MockGNPEModel()
    flow = GNPEFlowFactor(model, aliases={"ra": "ra@t_ref"})
    kernel = GNPEKernelFactor(model)
    gibbs = GibbsBlock(_DetectorTimesInit(), [kernel, flow], num_iterations=4)
    assert gibbs.conditioning == ["log_luminosity_distance_max"]
    chain = ChainComposer([DeltaFactor(PINS), gibbs, RAToEventFrame()])
    samples, lp = chain.sample_and_log_prob(12, context)
    assert lp is None
    assert samples["chirp_mass"].shape == (12,)
    assert torch.allclose(
        samples["log_luminosity_distance_max"], torch.full((12,), math.log(D_MAX))
    )
    assert len(model.contexts) == 4  # one network call per sweep
    for ctx in model.contexts:
        assert ctx.shape == (12, 2)
        assert torch.allclose(ctx[:, 1], torch.full((12,), PIN_STANDARDIZED))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class _StubModel:
    device = "cpu"

    def __init__(self, metadata):
        self.metadata = metadata
        self.base_metadata = metadata


def _init_metadata():
    metadata = copy.deepcopy(METADATA)
    data = metadata["train_settings"]["data"]
    data["inference_parameters"] = ["H1_time", "L1_time"]
    data["context_parameters"] = []
    del data["gnpe_time_shifts"], data["distance_prior_conditioning"]
    data["standardization"]["mean"].update(H1_time=0.0, L1_time=0.0)
    data["standardization"]["std"].update(H1_time=0.01, L1_time=0.01)
    return metadata


MAIN = _StubModel(METADATA)
INIT = _StubModel(_init_metadata())


def _step_names(sampler):
    return [type(step).__name__ for step in sampler.composer.steps]


def test_from_gnpe_models_roots_the_chain_in_the_pins():
    sampler = GWComposedSampler.from_gnpe_models(
        INIT, MAIN, _event_data(), num_iterations=2, fixed_context_parameters=PINS
    )
    assert _step_names(sampler)[:2] == ["DeltaFactor", "GibbsBlock"]
    assert sampler.composer.steps[0].values == PINS
    assert sampler.composer.steps[1].conditioning == ["log_luminosity_distance_max"]


def test_from_singlestep_gnpe_roots_the_chain_in_the_pins():
    proxies = DeltaFactor({"H1_time_proxy": 0.0, "L1_time_proxy": 0.0})
    sampler = GWComposedSampler.from_singlestep_gnpe(
        MAIN, proxies, _event_data(), fixed_context_parameters=PINS
    )
    assert _step_names(sampler)[:3] == ["DeltaFactor", "DeltaFactor", "GNPEFlowFactor"]
    assert sampler.composer.steps[0].values == PINS


@pytest.mark.parametrize("pins", [None, {"chirp_mass_proxy": 1.2}])
def test_gnpe_builders_require_exactly_the_pins(pins):
    with pytest.raises(ValueError, match="exactly these keys"):
        GWComposedSampler.from_gnpe_models(
            INIT, MAIN, _event_data(), num_iterations=1, fixed_context_parameters=pins
        )
    proxies = DeltaFactor({"H1_time_proxy": 0.0, "L1_time_proxy": 0.0})
    with pytest.raises(ValueError, match="exactly these keys"):
        GWComposedSampler.from_singlestep_gnpe(
            MAIN, proxies, _event_data(), fixed_context_parameters=pins
        )


@pytest.mark.parametrize("d_max", [100.0, 20000.0])
def test_gnpe_builders_reject_a_bound_outside_the_trained_range(d_max):
    pins = {"log_luminosity_distance_max": math.log(d_max)}
    with pytest.raises(ValueError, match="outside the trained range"):
        GWComposedSampler.from_gnpe_models(
            INIT, MAIN, _event_data(), num_iterations=1, fixed_context_parameters=pins
        )
