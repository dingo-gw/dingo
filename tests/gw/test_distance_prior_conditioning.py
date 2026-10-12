"""Prior conditioning on the upper bound of the luminosity-distance prior: the
training transform draws luminosity_distance from the base prior truncated at a
sampled bound (or log-uniformly, with probability log_uniform_fraction), and the
single-network sampler checks the pinned bound and warns when samples pile up at it."""

import numpy as np
import pandas as pd
import pytest
import torch
from bilby.core.prior import DeltaFunction
from scipy.stats import kstest

from dingo.core.inference.steps import DeltaFactor
from dingo.gw.inference.sampler import GWComposedSampler
from dingo.gw.transforms import DistancePriorConditioning

BASE_PRIOR = (
    "bilby.gw.prior.UniformComovingVolume(minimum=10.0, maximum=20000.0, "
    "name='luminosity_distance')"
)
HYPERPRIOR = "bilby.core.prior.LogUniform(minimum=50.0, maximum=20000.0)"


def _draw(transform, d_max, n=20000):
    """Batched draws of the transform with the bound fixed to d_max."""
    transform.hyperprior = DeltaFunction(d_max)
    sample = {"extrinsic_parameters": {"luminosity_distance": np.ones(n)}}
    return transform(sample)["extrinsic_parameters"]


@pytest.mark.parametrize("d_max", [300.0, 5000.0])
def test_draws_follow_truncated_base_prior(d_max):
    np.random.seed(0)
    transform = DistancePriorConditioning(BASE_PRIOR, HYPERPRIOR)
    out = _draw(transform, d_max)
    d = out["luminosity_distance"]
    assert np.allclose(out["log_luminosity_distance_max"], np.log(d_max))
    assert d.min() >= 10.0 and d.max() <= d_max * (1 + 1e-6)
    base = transform.base_prior
    p_value = kstest(d, lambda x: base.cdf(x) / base.cdf(d_max)).pvalue
    assert p_value > 0.01


def test_log_uniform_fraction_one_draws_log_uniformly():
    np.random.seed(0)
    transform = DistancePriorConditioning(BASE_PRIOR, HYPERPRIOR, 1.0)
    d = _draw(transform, 5000.0)["luminosity_distance"]
    p_value = kstest(np.log(d), "uniform", args=(np.log(10.0), np.log(500.0))).pvalue
    assert p_value > 0.01


def test_unbatched_sample_gives_floats_in_range():
    np.random.seed(0)
    transform = DistancePriorConditioning(BASE_PRIOR, HYPERPRIOR, 0.5)
    out = transform({"extrinsic_parameters": {"luminosity_distance": 1.0}})
    params = out["extrinsic_parameters"]
    assert isinstance(params["luminosity_distance"], float)
    assert 50.0 <= np.exp(params["log_luminosity_distance_max"]) <= 20000.0
    assert params["luminosity_distance"] <= np.exp(
        params["log_luminosity_distance_max"]
    )


@pytest.mark.parametrize(
    "hyperprior, fraction, match",
    [
        ("bilby.core.prior.LogUniform(minimum=50.0, maximum=10000.0)", 0.0, "equal"),
        ("bilby.core.prior.LogUniform(minimum=10.0, maximum=20000.0)", 0.0, "exceed"),
        (HYPERPRIOR, 1.5, "log_uniform_fraction"),
    ],
)
def test_invalid_settings_are_rejected(hyperprior, fraction, match):
    with pytest.raises(ValueError, match=match):
        DistancePriorConditioning(BASE_PRIOR, hyperprior, fraction)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_METADATA = {
    "dataset_settings": {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 256.0,
            "delta_f": 0.25,
        },
        "waveform_generator": {"approximant": "IMRPhenomXPHM", "f_ref": 20.0},
        "intrinsic_prior": {
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass("
            "minimum=20.0, maximum=40.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio("
            "minimum=0.5, maximum=1.0)",
        },
    },
    "train_settings": {
        "data": {
            "detectors": ["H1"],
            "ref_time": 1126259462.391,
            "extrinsic_prior": {"luminosity_distance": BASE_PRIOR},
            "inference_parameters": ["chirp_mass", "luminosity_distance"],
            "context_parameters": ["log_luminosity_distance_max"],
            "distance_prior_conditioning": {"luminosity_distance_max": HYPERPRIOR},
            "standardization": {
                "mean": {
                    "chirp_mass": 30.0,
                    "luminosity_distance": 3000.0,
                    "log_luminosity_distance_max": 7.0,
                },
                "std": {
                    "chirp_mass": 5.0,
                    "luminosity_distance": 3000.0,
                    "log_luminosity_distance_max": 1.5,
                },
            },
        }
    },
}


class _StubModel:
    metadata = _METADATA
    base_metadata = _METADATA
    device = "cpu"


def _event_data(n_bins=1025):
    return {
        "waveform": {"H1": np.ones(n_bins, dtype=complex)},
        "asds": {"H1": np.ones(n_bins)},
    }


@pytest.mark.parametrize("d_max", [50.0, 1000.0, 20000.0])
def test_from_model_pins_bound_in_trained_range(d_max):
    pins = {"log_luminosity_distance_max": np.log(d_max)}
    sampler = GWComposedSampler.from_model(_StubModel(), _event_data(), None, pins)
    assert isinstance(sampler.composer.steps[0], DeltaFactor)


@pytest.mark.parametrize("d_max", [20.0, 30000.0])
def test_from_model_rejects_bound_outside_trained_range(d_max):
    pins = {"log_luminosity_distance_max": np.log(d_max)}
    with pytest.raises(ValueError, match="outside the trained range"):
        GWComposedSampler.from_model(_StubModel(), _event_data(), None, pins)


class _StubComposer:
    """Returns fixed samples at a bound of 1000 Mpc."""

    def __init__(self, luminosity_distance):
        self.luminosity_distance = torch.tensor(luminosity_distance)

    def sample(self, num_samples, context, batch_size):
        n = len(self.luminosity_distance)
        return {
            "luminosity_distance": self.luminosity_distance,
            "log_luminosity_distance_max": torch.full((n,), np.log(1000.0)),
        }


def test_run_sampler_warns_when_samples_pile_up_at_bound():
    sampler = GWComposedSampler(_StubComposer(np.linspace(500.0, 999.0, 100)), None)
    with pytest.warns(UserWarning, match="truncated"):
        sampler.run_sampler(100)


def test_run_sampler_is_quiet_below_bound(recwarn):
    sampler = GWComposedSampler(_StubComposer(np.linspace(100.0, 800.0, 100)), None)
    samples = sampler.run_sampler(100)
    assert isinstance(samples, pd.DataFrame)
    assert not [w for w in recwarn if "truncated" in str(w.message)]
