"""CI unit tests for SyntheticPhaseFactor (and a GWSamplerContext helper), using a mock
context / likelihood so no waveform models or LAL calls are needed. End-to-end parity
against Result.sample_proposal_extensions is covered by the model-based harness."""

import numpy as np
import pytest
import torch
from bilby.core.utils import random as bilby_random

from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import SyntheticPhaseFactor, SyntheticPhasePsiFactor
from dingo.gw.likelihood import StationaryGaussianGWLikelihood


class _MockLikelihood:
    """Exposes the methods the factor uses, deterministic in `chirp_mass`."""

    def __init__(self):
        self.phase_grid = None

    def d_inner_h_complex_multi(self, theta, num_processes=1):
        # (2, 2)-approx path: one complex overlap (d | h) per row.
        return np.array([complex(cm, 0.5) for cm in theta["chirp_mass"].to_numpy()])

    def phase_grid_terms(self, theta, psi_dependent=False):
        # exact path: a single m = 1 mode with (d | mu_1) = chirp_mass, so that
        # log L(phase) = chirp_mass * cos(phase). With the psi basis, the second
        # projection has (d | mu_1) = chirp_mass / 2, so that
        # log L(phase, psi) = chirp_mass * cos(phase) * (cos 2psi + sin 2psi / 2).
        K = 2 if psi_dependent else 1
        kappa = complex(theta["chirp_mass"]) * np.array([1.0, 0.5])[:K]
        return {
            "m_vals": np.array([1]),
            "kappa2_modes": kappa[:, None],
            "rho2opt_const": np.zeros((K, K)),
            "deltas": np.array([], dtype=int),
            "rho2opt_crossterms": np.zeros((K, K, 0), dtype=complex),
        }

    log_Zn = 0.0
    log_likelihood_from_phase_grid_terms = (
        StationaryGaussianGWLikelihood.log_likelihood_from_phase_grid_terms
    )

    @staticmethod
    def log_l_phase_psi(chirp_mass, phase, psi):
        """The mock's log L(phase, psi) in closed form."""
        return chirp_mass * np.cos(phase) * (np.cos(2 * psi) + 0.5 * np.sin(2 * psi))


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
    assert factor.produces == ["phase", "log_likelihood"]
    _seed(2)
    block_cached, log_prob_cached = factor.sample_and_log_prob(1, context, given)

    phase = block_cached["phase"].numpy()
    assert np.array_equal(phase, block["phase"].numpy())
    assert np.array_equal(log_prob_cached.numpy(), log_prob.numpy())
    expected = np.cos(phase) * given["chirp_mass"].numpy()
    assert block_cached["log_likelihood"].dtype == torch.float64
    assert np.allclose(block_cached["log_likelihood"].numpy(), expected)


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
    assert set(out) == {"chirp_mass", "phase", "log_likelihood"}


def test_cached_log_likelihood_requires_exact_mode():
    with pytest.raises(ValueError, match="exact mode"):
        SyntheticPhaseFactor(
            conditioning=["chirp_mass"],
            approximation_22_mode=True,
            cache_log_likelihood=True,
        )


# ---------------------------------------------------------------------------
# SyntheticPhasePsiFactor
# ---------------------------------------------------------------------------


def _psi_factor(n_grid=65, n_grid_psi=33, **kwargs):
    return SyntheticPhasePsiFactor(
        conditioning=["chirp_mass"], n_grid=n_grid, n_grid_psi=n_grid_psi, **kwargs
    )


def test_phase_psi_factor_produces_both_angles():
    n = 6
    factor = _psi_factor()
    assert factor.parameters == ["phase", "psi"]
    assert factor.produces == ["phase", "psi"]
    _seed(0)
    block, log_prob = factor.sample_and_log_prob(1, _MockContext(), _given(n))
    phase, psi = block["phase"].numpy(), block["psi"].numpy()
    assert set(block) == {"phase", "psi"}
    assert phase.shape == psi.shape == log_prob.shape == (n,)
    assert (phase >= 0).all() and (phase <= 2 * np.pi).all()
    assert (psi >= 0).all() and (psi <= np.pi).all()
    assert np.isfinite(log_prob.numpy()).all()
    with pytest.raises(ValueError):
        factor.sample_and_log_prob(2, _MockContext(), _given(n))


def test_phase_psi_profile_marginal_matches_formula():
    # q(phase) is the psi-marginal of exp(log L) over the psi grid without its
    # duplicated endpoint psi = pi (plus the floor), computed here in chunks of samples.
    n, weight = 5, 0.01
    factor = _psi_factor(uniform_weight=weight)
    factor.max_grid_elements = 2 * factor.n_grid * factor.n_grid_psi  # 2 rows/chunk
    given = _given(n)
    _, _, phases, profile = factor._phase_profile(given, _MockContext())
    psis = np.linspace(0, np.pi, factor.n_grid_psi)
    cm = given["chirp_mass"].numpy()
    log_l = _MockLikelihood.log_l_phase_psi(
        cm[:, None, None], phases[None, :, None], psis[None, None, :]
    )
    expected = np.exp(log_l[..., :-1]).sum(axis=-1)
    expected /= expected.max(axis=1, keepdims=True)
    expected += expected.mean(axis=1, keepdims=True) * weight
    assert profile.shape == (n, factor.n_grid)
    assert np.allclose(profile, expected)


def test_phase_psi_log_prob_replug_matches_sample():
    # log_prob at the drawn (phase, psi) equals the sampled log q: the psi
    # conditional is rebuilt exactly at the given phase, as in the draw.
    n = 6
    factor = _psi_factor()
    context, given = _MockContext(), _given(n)
    _seed(1)
    block, log_prob = factor.sample_and_log_prob(1, context, given)
    replug = factor.log_prob(
        {"phase": block["phase"], "psi": block["psi"]}, context, given
    )
    assert np.allclose(log_prob.numpy(), replug.numpy())


def test_phase_psi_density_is_normalized():
    # exp(log q(phase, psi)) integrates to 1 over [0, 2pi) x [0, pi) for one sample.
    factor = _psi_factor(n_grid=129, n_grid_psi=65)
    phase_mesh, psi_mesh = np.meshgrid(
        np.linspace(0, 2 * np.pi, 121), np.linspace(0, np.pi, 61), indexing="ij"
    )
    n = phase_mesh.size
    given = {"chirp_mass": torch.full((n,), 3.0, dtype=torch.float64)}
    log_q = factor.log_prob(
        {
            "phase": torch.as_tensor(phase_mesh.ravel()),
            "psi": torch.as_tensor(psi_mesh.ravel()),
        },
        _MockContext(),
        given,
    ).numpy()
    q = np.exp(log_q).reshape(phase_mesh.shape)
    integral = np.trapezoid(np.trapezoid(q, psi_mesh[0], axis=1), phase_mesh[:, 0])
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_phase_psi_cached_log_likelihood_at_drawn_point():
    n = 6
    context, given = _MockContext(), _given(n)
    _seed(2)
    block, log_prob = _psi_factor().sample_and_log_prob(1, context, given)
    factor = _psi_factor(cache_log_likelihood=True)
    assert factor.produces == ["phase", "psi", "log_likelihood"]
    _seed(2)
    block_cached, log_prob_cached = factor.sample_and_log_prob(1, context, given)
    phase, psi = block_cached["phase"].numpy(), block_cached["psi"].numpy()
    assert np.array_equal(phase, block["phase"].numpy())
    assert np.array_equal(psi, block["psi"].numpy())
    assert np.array_equal(log_prob_cached.numpy(), log_prob.numpy())
    expected = _MockLikelihood.log_l_phase_psi(given["chirp_mass"].numpy(), phase, psi)
    assert np.allclose(block_cached["log_likelihood"].numpy(), expected)


def test_phase_psi_cached_log_likelihood_is_chain_output():
    from dingo.core.inference.composer import ChainComposer
    from dingo.core.inference.steps import SampleTableFactor

    table = SampleTableFactor({"chirp_mass": np.linspace(20.0, 40.0, 4)})
    _seed(3)
    context = _MockContext()
    context.device = None
    out, _ = ChainComposer(
        [table, _psi_factor(cache_log_likelihood=True)]
    ).sample_and_log_prob(1, context)
    assert set(out) == {"chirp_mass", "phase", "psi", "log_likelihood"}
