"""
Tests for StationaryGaussianGWLikelihood.log_likelihood_phase_grid().
"""

import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.likelihood import StationaryGaussianGWLikelihood, inner_product
from dingo.gw.waveform_generator import sum_contributions_m

THETA = {
    "mass_1": 45.0,
    "mass_2": 33.0,
    "a_1": 0.5,
    "a_2": 0.3,
    "tilt_1": 1.1,
    "tilt_2": 0.6,
    "phi_12": 1.9,
    "phi_jl": 4.2,
    "luminosity_distance": 800.0,
    "theta_jn": 1.0,
    "geocent_time": 0.0,
    "phase": 0.0,
    "ra": 1.3,
    "dec": -0.4,
    "psi": 2.0,
}


@pytest.fixture
def likelihood():
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=1 / 4.0)

    # Noise realisation as data, flat ASD. As elsewhere in the test suite, values
    # below f_min are zeroed (data) and set to 1 (ASD).
    rng = np.random.default_rng(42)
    waveform, asds = {}, {}
    for ifo in ["H1", "L1"]:
        d = (rng.normal(size=len(domain)) + 1j * rng.normal(size=len(domain))) * 1e-23
        waveform[ifo] = np.where(domain.frequency_mask, d, 0.0)
        asds[ifo] = np.where(domain.frequency_mask, 1e-23, 1.0)

    return StationaryGaussianGWLikelihood(
        wfg_kwargs={
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            # Required by the phase grid: the cartesian spins must not be
            # rederived at each phase.
            "spin_conversion_phase": 0.0,
        },
        wfg_domain=domain,
        data_domain=domain,
        event_data={"waveform": waveform, "asds": asds},
        t_ref=1126259462.4,
    )


def test_phase_grid_matches_direct_evaluation(likelihood):
    """The grid reproduces log L = log_Zn + (d, mu) - (mu, mu) / 2 evaluated phase
    by phase from the same m-components.

    Both sides start from one call to signal_m(), so the waveform model cancels
    exactly and only the vectorised algebra is under test -- its broadcasting
    shapes, its reduction axis, and the sign of every exp(-i * m * phase).
    """
    phases = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    grid = likelihood.log_likelihood_phase_grid(THETA, phases=phases)

    pol_m = {
        m: pol["waveform"]
        for m, pol in likelihood.signal_m({**THETA, "phase": 0}).items()
    }
    d = likelihood.whitened_strains
    min_idx = likelihood.data_domain.min_idx

    reference = []
    for phase in phases:
        mu = sum_contributions_m(pol_m, phase_shift=phase)
        rho2opt = sum(inner_product(m, m, min_idx) for m in mu.values())
        kappa2 = sum(
            inner_product(d_ifo, mu_ifo, min_idx)
            for d_ifo, mu_ifo in zip(d.values(), mu.values())
        )
        reference.append(likelihood.log_Zn + kappa2 - rho2opt / 2)

    assert grid.shape == phases.shape
    np.testing.assert_allclose(grid, reference, rtol=1e-9)


def test_phase_grid_is_2pi_periodic(likelihood):
    """phase and phase + 2 * pi describe the same waveform."""
    phases = np.array([0.4, 2.7, 5.5])
    np.testing.assert_allclose(
        likelihood.log_likelihood_phase_grid(THETA, phases=phases + 2 * np.pi),
        likelihood.log_likelihood_phase_grid(THETA, phases=phases),
        rtol=1e-9,
    )


def test_terms_reproduce_direct_likelihood_at_off_grid_phase(likelihood):
    """The likelihood from phase_grid_terms() at arbitrary phases equals the direct
    likelihood there, including a calibration curve. This is what lets importance
    sampling reuse the value cached by the synthetic phase."""
    rng = np.random.default_rng(0)
    calibration = {
        f"recalib_{ifo}_{q}_{i}": rng.normal(scale=0.05)
        for ifo in ["H1", "L1"]
        for q in ["amplitude", "phase"]
        for i in range(5)
    }
    phases = np.array([0.37, 2.9, 5.81])
    for extra in ({}, calibration):
        theta = {**THETA, **extra}
        terms = likelihood.phase_grid_terms(theta)
        from_terms = likelihood.log_likelihood_from_phase_grid_terms(terms, phases)
        direct = [likelihood.log_likelihood({**theta, "phase": p}) for p in phases]
        np.testing.assert_allclose(from_terms, direct, rtol=1e-9)
    # The calibration curve changes the likelihood.
    assert not np.allclose(
        from_terms,
        likelihood.log_likelihood_from_phase_grid_terms(
            likelihood.phase_grid_terms(THETA), phases
        ),
    )


def test_stacked_terms_match_per_sample_evaluation(likelihood):
    """Terms of several samples, stacked along a batch dimension, evaluate to the
    per-sample results, on a shared grid and at one phase per sample."""
    terms_per_sample = [
        likelihood.phase_grid_terms({**THETA, "mass_1": m1}) for m1 in (45.0, 50.0)
    ]
    terms = {
        "m_vals": terms_per_sample[0]["m_vals"],
        "deltas": terms_per_sample[0]["deltas"],
        **{
            k: np.array([t[k] for t in terms_per_sample])
            for k in ("kappa2_modes", "rho2opt_const", "rho2opt_crossterms")
        },
    }
    grid = np.linspace(0, 2 * np.pi, 7)
    drawn = np.array([0.3, 4.2])
    on_grid = likelihood.log_likelihood_from_phase_grid_terms(terms, grid)
    at_drawn = likelihood.log_likelihood_from_phase_grid_terms(terms, drawn[:, None])
    assert on_grid.shape == (2, 7) and at_drawn.shape == (2, 1)
    for i, t in enumerate(terms_per_sample):
        np.testing.assert_allclose(
            on_grid[i], likelihood.log_likelihood_from_phase_grid_terms(t, grid)
        )
        np.testing.assert_allclose(
            at_drawn[i],
            likelihood.log_likelihood_from_phase_grid_terms(t, drawn[i : i + 1]),
        )


def test_psi_dependent_terms_reproduce_direct_likelihood(likelihood):
    """log L from psi-dependent terms at off-grid (phase, psi) equals the direct
    likelihood, so both s(psi) = cos2psi s(0) + sin2psi s(pi/4) and the mode algebra
    hold through the real projection pipeline."""
    phases = np.array([0.37, 2.9, 5.81])
    psis = np.array([0.1, 1.2, 2.8])
    terms = likelihood.phase_grid_terms(THETA, psi_dependent=True)
    grid = likelihood.log_likelihood_from_phase_grid_terms(terms, phases, psis)
    direct = [
        [likelihood.log_likelihood({**THETA, "phase": p, "psi": s}) for s in psis]
        for p in phases
    ]
    assert grid.shape == (3, 3)
    np.testing.assert_allclose(grid, direct, rtol=1e-9)


def test_psi_dependent_terms_reduce_to_phase_only(likelihood):
    """At the sample's psi, the K = 2 grid equals the K = 1 phase grid."""
    phases = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    theta = {**THETA, "psi": 0.7}
    grid = likelihood.log_likelihood_from_phase_grid_terms(
        likelihood.phase_grid_terms(theta, psi_dependent=True), phases, np.array([0.7])
    )[:, 0]
    np.testing.assert_allclose(
        grid, likelihood.log_likelihood_phase_grid(theta, phases=phases), rtol=1e-9
    )


def test_stacked_psi_dependent_terms_at_one_point_per_sample(likelihood):
    """Stacked K = 2 terms evaluate at one (phase, psi) per sample, shape (N, 1, 1).
    This is what the synthetic phase+psi cache uses."""
    thetas = [{**THETA, "mass_1": m1} for m1 in (45.0, 50.0)]
    per = [likelihood.phase_grid_terms(t, psi_dependent=True) for t in thetas]
    terms = {
        "m_vals": per[0]["m_vals"],
        "deltas": per[0]["deltas"],
        **{
            k: np.array([t[k] for t in per])
            for k in ("kappa2_modes", "rho2opt_const", "rho2opt_crossterms")
        },
    }
    phase, psi = np.array([[1.0], [4.0]]), np.array([[0.3], [2.0]])
    out = likelihood.log_likelihood_from_phase_grid_terms(terms, phase, psi)
    direct = [
        likelihood.log_likelihood({**t, "phase": p[0], "psi": s[0]})
        for t, p, s in zip(thetas, phase, psi)
    ]
    assert out.shape == (2, 1, 1)
    np.testing.assert_allclose(out[:, 0, 0], direct, rtol=1e-9)
    # A shared (phase, psi) grid gives (N, G, H).
    grid = likelihood.log_likelihood_from_phase_grid_terms(
        terms, np.linspace(0, 2 * np.pi, 5), np.linspace(0, np.pi, 3)
    )
    assert grid.shape == (2, 5, 3)


def test_psis_must_match_basis(likelihood):
    """psis is required for psi-dependent terms and rejected otherwise."""
    phases = np.array([0.1])
    with pytest.raises(ValueError):
        likelihood.log_likelihood_from_phase_grid_terms(
            likelihood.phase_grid_terms(THETA), phases, np.array([0.2])
        )
    with pytest.raises(ValueError):
        likelihood.log_likelihood_from_phase_grid_terms(
            likelihood.phase_grid_terms(THETA, psi_dependent=True), phases
        )


def test_psi_dependent_algebra_matches_brute_force():
    """The (phase, psi) grid algebra, checked without a waveform model: terms built
    from random per-mode strains A_m (psi = 0) and B_m (psi = pi/4) and random data
    reproduce the brute-force log Zn + Re<d, mu> - <mu, mu> / 2 with
    mu(phase, psi) = sum_m exp(-i m phase) [cos 2psi A_m + sin 2psi B_m]."""
    rng = np.random.default_rng(0)
    ifos, n_bins, m_vals = ["H1", "L1"], 40, list(range(-3, 4))

    def strain():
        return rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)

    basis = [{m: {ifo: strain() for ifo in ifos} for m in m_vals} for _ in range(2)]
    d = {ifo: strain() for ifo in ifos}
    log_Zn = -0.5 * sum(inner_product(d[ifo], d[ifo]) for ifo in ifos)

    def overlap(x, y):
        return sum(np.sum(x[ifo].conj() * y[ifo]) for ifo in ifos)

    from collections import defaultdict

    crossterms = defaultdict(lambda: np.zeros((2, 2), dtype=complex))
    for i, m in enumerate(m_vals):
        for n in m_vals[i + 1 :]:
            pair = np.array([[overlap(b[m], c[n]) for c in basis] for b in basis])
            crossterms[n - m] += pair + pair.T
    deltas = sorted(crossterms)
    terms = {
        "m_vals": np.array(m_vals),
        "deltas": np.array(deltas),
        "kappa2_modes": np.array([[overlap(d, b[m]) for m in m_vals] for b in basis]),
        "rho2opt_const": np.array(
            [
                [sum(overlap(b[m], c[m]).real for m in m_vals) for c in basis]
                for b in basis
            ]
        ),
        "rho2opt_crossterms": np.moveaxis(
            np.array([crossterms[delta] for delta in deltas]), 0, -1
        ),
    }

    class _Terms:
        pass

    fake = _Terms()
    fake.log_Zn = log_Zn
    phases = np.linspace(0, 2 * np.pi, 9)
    psis = np.linspace(0, np.pi, 6)
    fast = StationaryGaussianGWLikelihood.log_likelihood_from_phase_grid_terms(
        fake, terms, phases, psis
    )

    brute = np.zeros((len(phases), len(psis)))
    for i, phase in enumerate(phases):
        for j, psi in enumerate(psis):
            mu = {
                ifo: sum(
                    np.exp(-1j * m * phase)
                    * (
                        np.cos(2 * psi) * basis[0][m][ifo]
                        + np.sin(2 * psi) * basis[1][m][ifo]
                    )
                    for m in m_vals
                )
                for ifo in ifos
            }
            brute[i, j] = log_Zn + overlap(d, mu).real - 0.5 * overlap(mu, mu).real
    np.testing.assert_allclose(fast, brute, rtol=1e-10, atol=1e-10)
