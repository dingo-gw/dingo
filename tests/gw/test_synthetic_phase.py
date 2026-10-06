"""CI unit tests for SyntheticPhaseFactor (and a GWSamplerContext helper), using a mock
context / likelihood so no waveform models or LAL calls are needed. End-to-end parity
against Result.sample_proposal_extensions is covered by the model-based harness."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from bilby.core.utils import random as bilby_random

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import SyntheticPhaseFactor
from dingo.gw.likelihood import StationaryGaussianGWLikelihood


class _MockLikelihood:
    """Exposes the methods the factor uses, deterministic in `chirp_mass`."""

    def __init__(self, uses_dft_phase_decomposition=True, phase_is_global_factor=True):
        self.phase_grid = None
        self.waveform_generator = SimpleNamespace(
            uses_dft_phase_decomposition=uses_dft_phase_decomposition,
            phase_is_global_factor=phase_is_global_factor,
            spin_conversion_phase=0.0,
            approximant_str="mock",
        )

    def phase_grid_terms_22(self, theta):
        # (2, 2) path: the overlap (d | h_0) and the norm (h_0 | h_0) of one row.
        return {
            "d_inner_h": complex(theta["chirp_mass"], 0.5),
            "h_inner_h": 2.0 * float(theta["chirp_mass"]),
        }

    def phase_grid_terms(self, theta):
        # exact path: a single m = 1 mode with (d | mu_1) = chirp_mass, so that
        # log L(phase) = chirp_mass * cos(phase).
        return {
            "m_vals": np.array([1]),
            "kappa2_modes": np.array([complex(theta["chirp_mass"])]),
            "rho2opt_const": 0.0,
            "deltas": np.array([], dtype=int),
            "rho2opt_crossterms": np.array([], dtype=complex),
        }

    log_Zn = 0.0
    log_likelihood_from_phase_grid_terms = (
        StationaryGaussianGWLikelihood.log_likelihood_from_phase_grid_terms
    )
    log_likelihood_22_from_terms = (
        StationaryGaussianGWLikelihood.log_likelihood_22_from_terms
    )


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
    phases, profile, _ = factor._phase_profile(given, _MockContext())

    cm = given["chirp_mass"].numpy()
    kappa = np.array([complex(c, 0.5) for c in cm])
    log_posterior = np.outer(kappa, np.exp(2j * phases)).real - cm[:, None]
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
    phases, profile, _ = factor._phase_profile(_given(n), _MockContext())
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


def test_factor_builds_likelihood_from_context():
    # The factor takes the likelihood from the context, passing its base-domain
    # choice (the same one importance sampling evaluates with) and any waveform
    # generator overrides.
    recorded = []

    class _RecordingContext(_MockContext):
        def likelihood(self, **kwargs):
            recorded.append(kwargs)
            return super().likelihood()

    for use_base_domain in (False, True):
        factor = SyntheticPhaseFactor(
            conditioning=["chirp_mass"],
            n_grid=11,
            approximation_22_mode=True,
            use_base_domain=use_base_domain,
        )
        _seed()
        factor.sample_and_log_prob(1, _RecordingContext(), _given())
        assert recorded[-1] == {
            "use_base_domain": use_base_domain,
            "wfg_updates": None,
        }

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=11,
        approximation_22_mode=True,
        wfg_updates={"use_dft_phase_decomposition": False},
    )
    _seed()
    factor.sample_and_log_prob(1, _RecordingContext(), _given())
    assert recorded[-1] == {
        "use_base_domain": False,
        "wfg_updates": {"use_dft_phase_decomposition": False},
    }


def test_cached_log_likelihood_at_drawn_phase():
    # With cache_log_likelihood, the factor also emits log L at the drawn (off-grid)
    # phase, evaluated from the terms rather than read off the grid; the phase draw
    # and its log q are unchanged.
    n = 6
    kwargs = dict(conditioning=["chirp_mass"], n_grid=33, approximation_22_mode=False)
    context, given = _MockContext(), _given(n)
    _seed(2)
    block, log_prob = SyntheticPhaseFactor(**kwargs).sample_and_log_prob(
        1, context, given
    )
    factor = SyntheticPhaseFactor(**kwargs, cache_log_likelihood=True)
    assert factor.produces == ["phase", "log_likelihood_cache"]
    _seed(2)
    block_cached, log_prob_cached = factor.sample_and_log_prob(1, context, given)

    phase = block_cached["phase"].numpy()
    assert np.array_equal(phase, block["phase"].numpy())
    assert np.array_equal(log_prob_cached.numpy(), log_prob.numpy())
    expected = np.cos(phase) * given["chirp_mass"].numpy()
    assert block_cached["log_likelihood_cache"].dtype == torch.float64
    assert np.allclose(block_cached["log_likelihood_cache"].numpy(), expected)


def test_cached_log_likelihood_is_chain_output():
    from dingo.core.inference.composer import ChainComposer
    from dingo.core.inference.steps import SampleTableFactor

    table = SampleTableFactor({"chirp_mass": np.linspace(20.0, 40.0, 4)})
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=33,
        approximation_22_mode=False,
        cache_log_likelihood=True,
    )
    _seed(3)
    context = _MockContext()
    context.device = None
    out, _ = ChainComposer([table, factor]).sample_and_log_prob(1, context)
    assert set(out) == {"chirp_mass", "phase", "log_likelihood_cache"}


def test_cached_log_likelihood_requires_dft_phase_decomposition():
    # Without the DFT decomposition the m-components do not sum to exactly the
    # waveform of a direct likelihood call, so caching raises rather than bias IS.
    class _NoDFTContext(_MockContext):
        def likelihood(self, **kwargs):
            return _MockLikelihood(uses_dft_phase_decomposition=False)

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=33,
        approximation_22_mode=False,
        cache_log_likelihood=True,
    )
    with pytest.raises(ValueError, match="DFT phase decomposition"):
        factor.sample_and_log_prob(1, _NoDFTContext(), _given())


def test_cached_log_likelihood_requires_an_exact_22_path():
    # The (2, 2) likelihood equals a direct call only if a phase shift multiplies the
    # waveform by exp(2i phase); otherwise caching raises rather than bias IS.
    class _NoGlobalFactorContext(_MockContext):
        def likelihood(self, **kwargs):
            return _MockLikelihood(phase_is_global_factor=False)

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=33,
        approximation_22_mode=True,
        cache_log_likelihood=True,
    )
    with pytest.raises(ValueError, match="global exp"):
        factor.sample_and_log_prob(1, _NoGlobalFactorContext(), _given())


def test_cached_log_likelihood_refuses_a_nan_probe():
    # A model listed as having a global exp(2i phase) factor but whose waveform comes
    # back NaN must fail the probe, not slip through the "mismatch > tolerance" test.
    class _NaNContext(_MockContext):
        def likelihood(self, **kwargs):
            likelihood = _MockLikelihood()
            likelihood.waveform_generator.domain = UniformFrequencyDomain(
                20.0, 27.0, 1.0
            )
            likelihood.waveform_generator.generate_hplus_hcross = lambda theta: {
                "h_plus": np.full(8, np.nan, dtype=complex),
                "h_cross": np.full(8, np.nan, dtype=complex),
            }
            return likelihood

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid=33,
        approximation_22_mode=True,
        cache_log_likelihood=True,
    )
    with pytest.raises(ValueError, match="refusing to cache"):
        factor.sample_and_log_prob(1, _NaNContext(), _given())
