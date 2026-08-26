"""
GNPE chain steps on a mock main network: draws per proxy row, log_prob re-plug
without mutating the input, and the single-step chain with pinned or sampled
proxies (forward log_prob vs. the reverse fold).
"""

import math

import numpy as np
import pytest
import torch
from astropy.utils import iers

from dingo.core.inference.composer import ChainComposer
from dingo.core.inference.steps import DeltaFactor, Factor
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import (
    GNPEFlowFactor,
    GNPEKernelCorrection,
    GNPEKernelFactor,
    RAToEventFrame,
)

iers.conf.auto_download = False

INFERENCE_PARAMETERS = ["chirp_mass", "mass_ratio", "geocent_time", "ra", "dec"]
CONTEXT_PARAMETERS = ["L1_time_proxy_relative"]
REF_TIME = 1126259462.4
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
            },
            "inference_parameters": INFERENCE_PARAMETERS,
            "context_parameters": CONTEXT_PARAMETERS,
            "gnpe_time_shifts": {
                "kernel": "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)",
                "exact_equiv": True,
            },
            "standardization": {
                "mean": {
                    "chirp_mass": 28.0,
                    "mass_ratio": 0.6,
                    "geocent_time": 0.0,
                    "ra": 3.1,
                    "dec": 0.0,
                    "L1_time_proxy_relative": 0.0,
                },
                "std": {
                    "chirp_mass": 2.0,
                    "mass_ratio": 0.2,
                    "geocent_time": 0.01,
                    "ra": 0.3,
                    "dec": 0.3,
                    "L1_time_proxy_relative": 0.003,
                },
            },
        }
    },
}


class _MockGNPEModel:
    """A GNPE main network with an analytic density: standard normal in the standardized
    parameters, shifted by the conditioning, so log_prob re-plug is checkable."""

    metadata = METADATA
    base_metadata = METADATA
    device = "cpu"
    network = torch.nn.Identity()

    @staticmethod
    def _log_prob(z, ctx):
        return (
            -0.5 * (z**2).sum(-1)
            - 0.5 * z.shape[-1] * math.log(2 * math.pi)
            + 0.3 * ctx.sum(-1)
        )

    def sample_and_log_prob(self, data, ctx, num_samples=1):
        z = torch.randn(data.shape[0], num_samples, len(INFERENCE_PARAMETERS))
        return z, self._log_prob(z, ctx.unsqueeze(1))

    def log_prob(self, z, data, ctx):
        return self._log_prob(z, ctx)


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


def _factors():
    model = _MockGNPEModel()
    flow = GNPEFlowFactor(model, aliases={"ra": "ra@t_ref"})
    return flow, GNPEKernelFactor(model)


def _proxies(kernel, context, n):
    times = {"H1_time": 0.01 * torch.randn(n), "L1_time": 0.01 * torch.randn(n)}
    proxies, _ = kernel.sample_and_log_prob(1, context, times)
    return proxies


def test_log_prob_matches_sampling_and_leaves_the_input_untouched(context):
    torch.manual_seed(1)
    flow, kernel = _factors()
    proxies = _proxies(kernel, context, 50)
    out, lp = flow.sample_and_log_prob(1, context, proxies)
    theta_i = {p: out[p] for p in flow.parameters}
    before = theta_i["geocent_time"].clone()
    lp_1 = flow.log_prob(theta_i, context, proxies)
    lp_2 = flow.log_prob(theta_i, context, proxies)
    assert torch.allclose(lp_1, lp, atol=1e-5)
    # Regression: log_prob used to shift the caller's geocent_time in place.
    assert torch.equal(theta_i["geocent_time"], before)
    assert torch.equal(lp_1, lp_2)


def test_draws_num_samples_per_proxy_row(context):
    torch.manual_seed(2)
    flow, kernel = _factors()
    n_rows, k = 4, 3
    proxies = _proxies(kernel, context, n_rows)
    out, lp = flow.sample_and_log_prob(k, context, proxies)
    assert all(v.shape == (n_rows * k,) for v in out.values())
    assert lp.shape == (n_rows * k,)
    # The draws for proxy row i are rows k*i .. k*i+k-1: re-plug with the proxies
    # repeated accordingly.
    repeated = {p: v.repeat_interleave(k) for p, v in proxies.items()}
    theta_i = {p: out[p] for p in flow.parameters}
    assert torch.allclose(flow.log_prob(theta_i, context, repeated), lp, atol=1e-5)


def test_singlestep_chain_with_pinned_proxies(context):
    # A DeltaFactor proxy source does not draw, so the flow draws num_samples per
    # (single) pinned row; the reverse fold reproduces the forward log_prob.
    torch.manual_seed(3)
    flow, kernel = _factors()
    chain = ChainComposer(
        [
            DeltaFactor({"H1_time_proxy": 0.0, "L1_time_proxy": 0.0}),
            flow,
            GNPEKernelCorrection(kernel),
            RAToEventFrame(),
        ]
    )
    samples, lp = chain.sample_and_log_prob(20, context)
    assert lp.shape == (20,) and samples["ra"].shape == (20,)
    assert "delta_log_prob_target" in samples and "H1_time" not in samples
    assert torch.allclose(chain.log_prob(samples, context), lp, atol=1e-5)


class _ProxyRoot(Factor):
    """Gaussian proxies (a stand-in for the density-recovery NDE)."""

    parameters = ["H1_time_proxy", "L1_time_proxy"]
    conditioning = []

    def sample_and_log_prob(self, num_samples, context, given=None):
        z = 0.001 * torch.randn(num_samples, 2)
        samples = {"H1_time_proxy": z[:, 0], "L1_time_proxy": z[:, 1]}
        return samples, self.log_prob(samples, context)

    def log_prob(self, theta_i, context, given=None):
        z = torch.stack([theta_i["H1_time_proxy"], theta_i["L1_time_proxy"]], -1)
        z = z / 0.001
        return -0.5 * (z**2).sum(-1) - math.log(2 * math.pi) - 2 * math.log(0.001)


def test_singlestep_chain_with_sampled_proxies(context):
    torch.manual_seed(4)
    flow, kernel = _factors()
    chain = ChainComposer(
        [_ProxyRoot(), flow, GNPEKernelCorrection(kernel), RAToEventFrame()]
    )
    samples, lp = chain.sample_and_log_prob(20, context)
    assert lp.shape == (20,)
    assert torch.allclose(chain.log_prob(samples, context), lp, atol=1e-5)
