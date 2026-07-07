"""CI unit tests for SyntheticPhaseFactor (and a GWSamplerContext helper), using a mock
context / likelihood so no waveform models or LAL calls are needed. End-to-end parity
against Result.sample_synthetic_phase is covered by the model-based harness."""

import numpy as np
import pytest
import torch
from bilby.core.utils import random as bilby_random

from dingo.gw.inference.factors import GWSamplerContext, SyntheticPhaseFactor


class _MockLikelihood:
    """Exposes the two methods the factor uses, deterministic in `chirp_mass`."""

    def __init__(self):
        self.phase_grid = None

    def d_inner_h_complex_multi(self, theta, num_processes=1):
        # (2, 2)-approx path: one complex overlap (d | h) per row.
        return np.array([complex(cm, 0.5) for cm in theta["chirp_mass"].to_numpy()])

    def log_likelihood_phase_grid(self, theta):
        # exact path: a (n_grid,) log-likelihood per row (theta is a row dict).
        return np.cos(self.phase_grid) * float(theta["chirp_mass"])


class _MockContext:
    def likelihood(self, **kwargs):
        return _MockLikelihood()


def _given(n=5):
    return {"chirp_mass": torch.linspace(20.0, 40.0, n, dtype=torch.float64)}


def _seed(s=0):
    np.random.seed(s)
    bilby_random.seed(s)


def test_parameters_and_conditioning():
    factor = SyntheticPhaseFactor(conditioning=["chirp_mass", "theta_jn"])
    assert factor.parameters == ["phase"]
    assert factor.conditioning == ["chirp_mass", "theta_jn"]


def test_synthetic_phase_is_one_to_one():
    factor = SyntheticPhaseFactor(conditioning=["chirp_mass"])
    with pytest.raises(ValueError):
        factor.sample_and_log_prob(2, _MockContext(), _given())


def test_profile_approx_matches_formula():
    n, n_grid, weight = 5, 257, 0.01
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=n_grid,
        approximation_22_mode=True,
        uniform_weight=weight,
    )
    given = _given(n)
    phases, profile = factor._phase_profile(given, _MockContext())

    kappa = np.array([complex(cm, 0.5) for cm in given["chirp_mass"].numpy()])
    log_posterior = np.outer(kappa, np.exp(2j * phases)).real
    expected = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))
    expected += expected.mean(axis=1, keepdims=True) * weight

    assert phases.shape == (n_grid,)
    assert profile.shape == (n, n_grid)
    assert np.allclose(profile, expected)
    assert (profile > 0).all()  # uniform floor keeps it mass-covering


def test_profile_exact_mode_runs():
    n, n_grid = 4, 129
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"], n_grid=n_grid, approximation_22_mode=False
    )
    phases, profile = factor._phase_profile(_given(n), _MockContext())
    assert phases.shape == (n_grid,)
    assert profile.shape == (n, n_grid)
    assert (profile > 0).all()


def test_sample_shapes_and_range():
    n = 6
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"], approximation_22_mode=True
    )
    _seed(0)
    block, log_prob = factor.sample_and_log_prob(1, _MockContext(), _given(n))
    phase = block["phase"].numpy()
    assert set(block) == {"phase"}
    assert phase.shape == (n,) and log_prob.shape == (n,)
    assert (phase >= 0).all() and (phase <= 2 * np.pi).all()
    assert np.isfinite(log_prob.numpy()).all()


def test_log_prob_replug_matches_sample():
    # factor.log_prob at the drawn phase equals the sampled log q (same deterministic
    # profile, no re-draw).
    n = 6
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"], approximation_22_mode=True
    )
    context, given = _MockContext(), _given(n)
    _seed(1)
    block, log_prob = factor.sample_and_log_prob(1, context, given)
    log_prob_replug = factor.log_prob({"phase": block["phase"]}, context, given)
    assert np.allclose(log_prob.numpy(), log_prob_replug.numpy())


def test_context_frequency_override():
    def context(event_metadata):
        return GWSamplerContext(
            domain=None,
            data_prep=None,
            event_data={},
            event_metadata=event_metadata,
        )

    with_override = context({"minimum_frequency": 25.0})
    assert with_override._frequency("minimum_frequency", 20.0) == 25.0
    assert with_override._frequency("maximum_frequency", 1024.0) == 1024.0
    assert context(None)._frequency("minimum_frequency", 20.0) == 20.0


def test_factor_builds_likelihood_from_context():
    # The factor takes the likelihood from the context with no arguments -- the
    # data representation lives on the (possibly derived) context, not on the
    # factor.
    recorded = []

    class _RecordingContext(_MockContext):
        def likelihood(self, **kwargs):
            recorded.append(kwargs)
            return super().likelihood()

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=11,
        approximation_22_mode=True,
    )
    _seed()
    factor.sample_and_log_prob(1, _RecordingContext(), _given())
    assert recorded == [{}]
