"""CI unit tests for SyntheticPhaseFactor (and a GWSamplerContext helper), using a mock
context / likelihood so no waveform models or LAL calls are needed. End-to-end parity
against Result.sample_proposal_extensions is covered by the model-based harness."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from bilby.core.utils import random as bilby_random

from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import SyntheticPhaseFactor, SyntheticPhasePsiFactor
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
        # exact path: a single m = 1 mode with (d | mu_1) = chirp_mass at psi = 0 and
        # chirp_mass / 2 at psi = pi / 4, so that
        # log L(phase, psi) = chirp_mass * cos(phase) * (cos 2psi + sin 2psi / 2),
        # and log L(phase) = chirp_mass * cos(phase) at psi = 0.
        kappa = complex(theta["chirp_mass"]) * np.array([1.0, 0.5])
        return {
            "m_vals": np.array([1]),
            "kappa2_modes": kappa[:, None],
            "rho2opt_const": np.zeros((2, 2)),
            "deltas": np.array([], dtype=int),
            "rho2opt_crossterms": np.zeros((2, 2, 0), dtype=complex),
        }

    def phase_psi_terms_22(self, theta):
        # (2, 2) path at the basis angles psi = 0, pi / 4, see log_l_phase_psi_22.
        c = float(theta["chirp_mass"]) / 10
        return {
            "d_inner_h": c * np.array([1.0, 0.5j]),
            "h_inner_h": c * np.array([[1.0, 0.2], [0.2, 0.5]]),
        }

    log_Zn = 0.0
    log_likelihood_from_phase_grid_terms = (
        StationaryGaussianGWLikelihood.log_likelihood_from_phase_grid_terms
    )
    log_likelihood_22_from_terms = (
        StationaryGaussianGWLikelihood.log_likelihood_22_from_terms
    )
    terms_22_at_psi = StationaryGaussianGWLikelihood.terms_22_at_psi

    @staticmethod
    def log_l_phase_psi(chirp_mass, phase, psi):
        """The mock's log L(phase, psi) in closed form."""
        return chirp_mass * np.cos(phase) * (np.cos(2 * psi) + 0.5 * np.sin(2 * psi))

    @staticmethod
    def log_l_phase_psi_22(chirp_mass, phase, psi):
        """The mock's (2, 2) log L(phase, psi) in closed form."""
        c, x, y = chirp_mass / 10, np.cos(2 * psi), np.sin(2 * psi)
        return (c * (x + 0.5j * y) * np.exp(2j * phase)).real - c * (
            x**2 + 0.4 * x * y + 0.5 * y**2
        ) / 2


class _MockContext:
    def likelihood(self, **kwargs):
        return _MockLikelihood()


def _given(n=5):
    return {
        "chirp_mass": torch.linspace(20.0, 40.0, n, dtype=torch.float64),
        "psi": torch.zeros(n, dtype=torch.float64),
    }


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


def test_phase_22_density_is_the_floored_conditional():
    # On the (2, 2) path log q(phase) is exp(log L(phase)) normalized over the phase,
    # mixed with the uniform floor in the proportion of the grid densities, without
    # any phase grid; and the factor records no grid size there.
    n, weight = 4, 0.01
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"], approximation_22_mode=True, uniform_weight=weight
    )
    assert "n_grid_phase" not in factor.describe()
    given = _given(n)
    phases = np.linspace(0, 2 * np.pi, 4001)
    log_q = np.array(
        [
            factor.log_prob(
                {"phase": torch.full((n,), p, dtype=torch.float64)},
                _MockContext(),
                given,
            )
            for p in phases
        ]
    ).T
    cm = given["chirp_mass"].numpy()
    kappa = np.array([complex(c, 0.5) for c in cm])
    conditional = np.exp(np.outer(kappa, np.exp(2j * phases)).real)
    conditional /= np.trapezoid(conditional, phases, axis=1)[:, None]
    expected = (conditional + weight / (2 * np.pi)) / (1 + weight)
    np.testing.assert_allclose(np.exp(log_q), expected, rtol=1e-8)


def test_phase_22_draws_match_the_psi_factor():
    # With psi fixed, the phase factor's (2, 2) draw is the psi factor's conditional
    # draw: the same function, the same random numbers, the same phases.
    from dingo.gw.inference.steps import _sample_phase_22

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"], approximation_22_mode=True
    )
    given = _given(50)
    _seed(6)
    block, log_prob = factor.sample_and_log_prob(1, _MockContext(), given)
    _seed(6)
    z = np.array([complex(c, 0.5) for c in given["chirp_mass"].numpy()])
    np.testing.assert_array_equal(block["phase"].numpy(), _sample_phase_22(z, 0.01))
    replug = factor.log_prob({"phase": block["phase"]}, _MockContext(), given)
    np.testing.assert_allclose(log_prob.numpy(), replug.numpy())


def test_profile_exact_mode_runs():
    n, n_grid = 4, 129
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"], n_grid_phase=n_grid, approximation_22_mode=False
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
            n_grid_phase=11,
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
        n_grid_phase=11,
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
    kwargs = dict(
        conditioning=["chirp_mass"], n_grid_phase=33, approximation_22_mode=False
    )
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

    table = SampleTableFactor(
        {"chirp_mass": np.linspace(20.0, 40.0, 4), "psi": np.zeros(4)}
    )
    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass", "psi"],
        n_grid_phase=33,
        approximation_22_mode=False,
        cache_log_likelihood=True,
    )
    _seed(3)
    context = _MockContext()
    context.device = None
    out, _ = ChainComposer([table, factor]).sample_and_log_prob(1, context)
    assert set(out) == {"chirp_mass", "psi", "phase", "log_likelihood_cache"}


def test_cached_log_likelihood_requires_dft_phase_decomposition():
    # Without the DFT decomposition the m-components do not sum to exactly the
    # waveform of a direct likelihood call, so caching raises rather than bias IS.
    class _NoDFTContext(_MockContext):
        def likelihood(self, **kwargs):
            return _MockLikelihood(uses_dft_phase_decomposition=False)

    factor = SyntheticPhaseFactor(
        conditioning=["chirp_mass"],
        n_grid_phase=33,
        approximation_22_mode=False,
        cache_log_likelihood=True,
    )
    with pytest.raises(ValueError, match="DFT phase decomposition"):
        factor.sample_and_log_prob(1, _NoDFTContext(), _given())


# ---------------------------------------------------------------------------
# SyntheticPhasePsiFactor
# ---------------------------------------------------------------------------


def _psi_factor(n_grid_phase=65, n_grid_psi=33, **kwargs):
    return SyntheticPhasePsiFactor(
        conditioning=["chirp_mass"],
        n_grid_phase=n_grid_phase,
        n_grid_psi=n_grid_psi,
        **kwargs,
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
    factor.max_grid_elements = (
        2 * factor.n_grid_phase * factor.n_grid_psi
    )  # 2 rows/chunk
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
    assert profile.shape == (n, factor.n_grid_phase)
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


def test_phase_psi_conditional_follows_the_drawn_phase():
    # q(psi | phase) is built at the phase actually drawn, not at a grid point or a
    # fixed phase: the mock's psi profile changes shape with cos(phase), so an
    # evaluation at the wrong phase gives a different profile.
    factor = _psi_factor(n_grid_psi=33)
    given = _given(4)
    likelihood, terms, _, _ = factor._phase_profile(given, _MockContext())
    phase = np.array([0.3, 2.0, 3.4, 5.0])
    psis, profile = factor._psi_profile(likelihood, terms, phase)
    log_l = _MockLikelihood.log_l_phase_psi(
        given["chirp_mass"].numpy()[:, None], phase[:, None], psis[None, :]
    )
    expected = np.exp(log_l - log_l.max(axis=-1, keepdims=True))
    expected += expected.mean(axis=-1, keepdims=True) * factor.uniform_weight
    assert np.allclose(profile, expected)


def test_phase_psi_density_is_normalized():
    # exp(log q(phase, psi)) integrates to 1 over [0, 2pi) x [0, pi) for one sample.
    # q is piecewise constant on the grid cells, so integrate by the midpoint rule.
    factor = _psi_factor(n_grid_phase=129, n_grid_psi=65)
    n_phase, n_psi = 160, 80
    phase_mesh, psi_mesh = np.meshgrid(
        (np.arange(n_phase) + 0.5) * 2 * np.pi / n_phase,
        (np.arange(n_psi) + 0.5) * np.pi / n_psi,
        indexing="ij",
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
    integral = q.sum() * (2 * np.pi / n_phase) * (np.pi / n_psi)
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_phase_psi_cached_log_likelihood_at_drawn_point():
    n = 6
    context, given = _MockContext(), _given(n)
    _seed(2)
    block, log_prob = _psi_factor().sample_and_log_prob(1, context, given)
    factor = _psi_factor(cache_log_likelihood=True)
    assert factor.produces == ["phase", "psi", "log_likelihood_cache"]
    _seed(2)
    block_cached, log_prob_cached = factor.sample_and_log_prob(1, context, given)
    phase, psi = block_cached["phase"].numpy(), block_cached["psi"].numpy()
    assert np.array_equal(phase, block["phase"].numpy())
    assert np.array_equal(psi, block["psi"].numpy())
    assert np.array_equal(log_prob_cached.numpy(), log_prob.numpy())
    expected = _MockLikelihood.log_l_phase_psi(given["chirp_mass"].numpy(), phase, psi)
    assert np.allclose(block_cached["log_likelihood_cache"].numpy(), expected)


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
    assert set(out) == {"chirp_mass", "phase", "psi", "log_likelihood_cache"}


def _psi_factor_22(**kwargs):
    return _psi_factor(approximation_22_mode=True, **kwargs)


def test_phase_psi_22_psi_marginal_is_the_phase_integral():
    # On the (2, 2) path q(psi) integrates the phase out in closed form,
    # exp(-(h, h) / 2) I0(|(d, h)|): compare with a brute-force phase integral.
    n, weight = 5, 0.01
    factor = _psi_factor_22(uniform_weight=weight)
    given = _given(n)
    _, _, psis, profile = factor._psi_profile_22(given, _MockContext())
    phases = np.linspace(0, 2 * np.pi, 20001)[:-1]
    log_l = _MockLikelihood.log_l_phase_psi_22(
        given["chirp_mass"].numpy()[:, None, None],
        phases[None, :, None],
        psis[None, None, :],
    )
    expected = np.exp(log_l).mean(axis=1)
    expected /= expected.max(axis=1, keepdims=True)
    expected += expected.mean(axis=1, keepdims=True) * weight
    assert profile.shape == (n, factor.n_grid_psi)
    np.testing.assert_allclose(profile, expected, rtol=1e-10)


def test_phase_psi_22_phase_density_is_the_floored_conditional():
    # log q(phase | psi) is exp(log L(phase, psi)) normalized over the phase, mixed
    # with the uniform floor in the proportion of the grid densities.
    from dingo.gw.inference.steps import _log_prob_phase_22

    factor = _psi_factor_22()
    weight = factor.uniform_weight
    cm, psi = np.array([20.0, 40.0]), np.array([0.3, 2.2])
    terms = _MockLikelihood().terms_22_at_psi(
        {
            k: np.array([v1, v2])
            for (k, v1), v2 in zip(
                _MockLikelihood().phase_psi_terms_22({"chirp_mass": cm[0]}).items(),
                _MockLikelihood().phase_psi_terms_22({"chirp_mass": cm[1]}).values(),
            )
        },
        psi[:, None],
    )
    terms = {k: v[:, 0] for k, v in terms.items()}
    phases = np.linspace(0, 2 * np.pi, 4001)
    log_q = np.array(
        [_log_prob_phase_22(terms["d_inner_h"], np.full(2, p), weight) for p in phases]
    ).T
    conditional = np.exp(
        _MockLikelihood.log_l_phase_psi_22(cm[:, None], phases, psi[:, None])
    )
    conditional /= np.trapezoid(conditional, phases, axis=1)[:, None]
    expected = (conditional + weight / (2 * np.pi)) / (1 + weight)
    np.testing.assert_allclose(np.exp(log_q), expected, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(np.exp(log_q), phases, axis=1), 1.0)


def test_phase_psi_22_phase_draws_follow_their_density():
    # The phase drawn by _sample_phase_22 (von Mises in 2 phase, both halves, the
    # floor) is distributed as exp(log q(phase | psi)), for weak and strong
    # concentration.
    from scipy.stats import kstest
    from dingo.gw.inference.steps import _log_prob_phase_22, _sample_phase_22

    weight = 0.05
    n = 40000
    for z in (0.7 - 0.4j, 8.0 + 30.0j):
        _seed(4)
        phase = _sample_phase_22(np.full(n, z), weight)
        assert (phase >= 0).all() and (phase < 2 * np.pi).all()
        grid = np.linspace(0, 2 * np.pi, 200001)
        density = np.exp(_log_prob_phase_22(np.full(1, z), grid, weight))
        cdf = np.concatenate(
            [[0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(grid))]
        )
        assert cdf[-1] == pytest.approx(1.0, abs=1e-8)
        assert kstest(phase, lambda x: np.interp(x, grid, cdf)).pvalue > 1e-3


def test_phase_22_floor_fraction():
    # The floor replaces a fraction w / (1 + w) of the draws by uniform ones, the
    # proportion `_floored_density` gives the grid densities (not w). With a very
    # concentrated von Mises part, every draw away from its two modes is a floor draw.
    from dingo.gw.inference.steps import _sample_phase_22

    weight, n, half_width, mode = 0.5, 200000, 0.1, 0.4
    _seed(7)
    phase = _sample_phase_22(np.full(n, 1e6 * np.exp(-2j * mode)), weight)
    distance = np.abs(np.mod(phase - mode + np.pi / 2, np.pi) - np.pi / 2)
    away = np.mean(distance > half_width)
    expected = weight / (1 + weight) * (1 - 4 * half_width / (2 * np.pi))
    assert away == pytest.approx(expected, abs=5 * np.sqrt(expected / n))


def test_phase_psi_22_log_prob_replug_matches_sample():
    n = 6
    factor = _psi_factor_22()
    context, given = _MockContext(), _given(n)
    _seed(1)
    block, log_prob = factor.sample_and_log_prob(1, context, given)
    phase, psi = block["phase"].numpy(), block["psi"].numpy()
    assert phase.shape == psi.shape == log_prob.shape == (n,)
    assert (psi >= 0).all() and (psi <= np.pi).all()
    replug = factor.log_prob(
        {"phase": block["phase"], "psi": block["psi"]}, context, given
    )
    np.testing.assert_allclose(log_prob.numpy(), replug.numpy())


def test_phase_psi_22_density_is_normalized():
    # exp(log q(phase, psi)) integrates to 1 over [0, 2pi) x [0, pi) for one sample:
    # piecewise constant in psi, smooth in the phase.
    factor = _psi_factor_22(n_grid_psi=65)
    n_phase, n_psi = 160, 128
    phase_mesh, psi_mesh = np.meshgrid(
        (np.arange(n_phase) + 0.5) * 2 * np.pi / n_phase,
        (np.arange(n_psi) + 0.5) * np.pi / n_psi,
        indexing="ij",
    )
    n = phase_mesh.size
    given = {"chirp_mass": torch.full((n,), 30.0, dtype=torch.float64)}
    log_q = factor.log_prob(
        {
            "phase": torch.as_tensor(phase_mesh.ravel()),
            "psi": torch.as_tensor(psi_mesh.ravel()),
        },
        _MockContext(),
        given,
    ).numpy()
    q = np.exp(log_q).reshape(phase_mesh.shape)
    integral = q.sum() * (2 * np.pi / n_phase) * (np.pi / n_psi)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_phase_psi_22_cached_log_likelihood_at_drawn_point():
    # Caching passes the exactness probe and emits log L at the drawn pair, without
    # changing the draws or their log q.
    n = 6
    context, given = _MockContext(), _given(n)
    _seed(2)
    block, log_prob = _psi_factor_22().sample_and_log_prob(1, context, given)
    factor = _psi_factor_22(cache_log_likelihood=True)
    assert factor.produces == ["phase", "psi", "log_likelihood_cache"]
    assert "n_grid_phase" not in factor.describe()
    assert "n_grid_phase" in _psi_factor().describe()
    _seed(2)
    block_cached, log_prob_cached = factor.sample_and_log_prob(1, context, given)
    phase, psi = block_cached["phase"].numpy(), block_cached["psi"].numpy()
    assert np.array_equal(phase, block["phase"].numpy())
    assert np.array_equal(psi, block["psi"].numpy())
    assert np.array_equal(log_prob_cached.numpy(), log_prob.numpy())
    expected = _MockLikelihood.log_l_phase_psi_22(
        given["chirp_mass"].numpy(), phase, psi
    )
    np.testing.assert_allclose(block_cached["log_likelihood_cache"].numpy(), expected)


def test_phase_psi_22_pooled_draws_match_serial():
    # Every random number of the (2, 2) path is drawn in the parent process, so the
    # draws do not depend on the number of processes computing the terms.
    given = _given(60)
    out = []
    for num_processes in (1, 3):
        _seed(5)
        block, log_prob = _psi_factor_22(
            num_processes=num_processes
        ).sample_and_log_prob(1, _MockContext(), given)
        out.append((block["phase"].numpy(), block["psi"].numpy(), log_prob.numpy()))
    for serial, pooled in zip(*out):
        assert np.array_equal(serial, pooled)


@pytest.mark.parametrize("cls", [SyntheticPhaseFactor, SyntheticPhasePsiFactor])
def test_cached_log_likelihood_requires_an_exact_22_path(cls):
    # The (2, 2) likelihood equals a direct call only if a phase shift multiplies the
    # waveform by exp(2i phase); otherwise caching raises rather than bias IS.
    class _NoGlobalFactorContext(_MockContext):
        def likelihood(self, **kwargs):
            return _MockLikelihood(phase_is_global_factor=False)

    factor = cls(
        conditioning=["chirp_mass"],
        n_grid_phase=33,
        approximation_22_mode=True,
        cache_log_likelihood=True,
    )
    with pytest.raises(ValueError, match="global exp"):
        factor.sample_and_log_prob(1, _NoGlobalFactorContext(), _given())
