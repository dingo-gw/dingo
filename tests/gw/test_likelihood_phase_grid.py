"""
Tests for StationaryGaussianGWLikelihood.log_likelihood_phase_grid().
"""

import importlib.util

import numpy as np
import pytest

from dingo.gw.domains import MultibandedFrequencyDomain, UniformFrequencyDomain
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


# The cached-vs-direct comparison runs per decomposition path: the DFT phase
# decomposition of a LAL model on uniform and multibanded domains, and of
# SEOBNRv5PHM (needs pyseobnr >= 0.3.7).
@pytest.fixture(
    params=[
        "IMRPhenomXPHM-uniform",
        "IMRPhenomXPHM-multibanded",
        pytest.param(
            "SEOBNRv5PHM-uniform",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("pyseobnr") is None,
                reason="pyseobnr is not installed",
            ),
        ),
    ]
)
def likelihood(request):
    approximant, kind = request.param.split("-")
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=1 / 4.0)
    if kind == "multibanded":
        domain = MultibandedFrequencyDomain(
            nodes=[20.0, 40.0, 80.0, 512.0],
            delta_f_initial=1 / 4.0,
            base_domain=domain.domain_dict,
        )

    # Noise realisation as data, flat ASD. As elsewhere in the test suite, values
    # below f_min are zeroed (data) and set to 1 (ASD).
    rng = np.random.default_rng(42)
    waveform, asds = {}, {}
    for ifo in ["H1", "L1"]:
        d = (rng.normal(size=len(domain)) + 1j * rng.normal(size=len(domain))) * 1e-23
        waveform[ifo] = np.where(domain.frequency_mask, d, 0.0)
        asds[ifo] = np.where(domain.frequency_mask, 1e-23, 1.0)

    wfg_kwargs = {
        "approximant": approximant,
        "f_ref": 20.0,
        # Required by the phase grid: the cartesian spins must not be
        # rederived at each phase.
        "spin_conversion_phase": 0.0,
    }
    if approximant == "SEOBNRv5PHM":
        wfg_kwargs.update(new_interface=True, f_start=20.0)
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs=wfg_kwargs,
        wfg_domain=domain,
        data_domain=domain,
        event_data={"waveform": waveform, "asds": asds},
        t_ref=1126259462.4,
    )
    assert likelihood.waveform_generator.uses_dft_phase_decomposition
    return likelihood


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
    # The LAL models agree to round-off. For SEOBNRv5PHM, pyseobnr's multi-phase and
    # single-phase evaluations differ slightly (measured ~1e-5 nats).
    approximant = likelihood.waveform_generator.approximant_str
    atol = 1e-4 if approximant == "SEOBNRv5PHM" else 0.0
    for extra in ({}, calibration):
        theta = {**THETA, **extra}
        terms = likelihood.phase_grid_terms(theta)
        from_terms = likelihood.log_likelihood_from_phase_grid_terms(terms, phases)
        direct = [likelihood.log_likelihood({**theta, "phase": p}) for p in phases]
        np.testing.assert_allclose(from_terms, direct, rtol=1e-9, atol=atol)
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
            for k in ("kappa2_modes", "rho2opt_crossterms", "rho2opt_const")
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


def test_terms_22_reproduce_direct_likelihood():
    """The (2, 2) terms evaluate to the direct log likelihood at any phase, for a
    model whose phase shift is a global exp(2i phase) factor. This is what lets
    importance sampling reuse the value cached by the synthetic phase on the
    (2, 2) path, which is the only exact route for the NRTidal family (LALSimulation
    implements no frequency-domain modes for it)."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=1 / 4.0)
    rng = np.random.default_rng(42)
    waveform, asds = {}, {}
    for ifo in ["H1", "L1"]:
        d = (rng.normal(size=len(domain)) + 1j * rng.normal(size=len(domain))) * 1e-23
        waveform[ifo] = np.where(domain.frequency_mask, d, 0.0)
        asds[ifo] = np.where(domain.frequency_mask, 1e-23, 1.0)
    theta = {
        **THETA,
        "mass_1": 1.6,
        "mass_2": 1.3,
        "lambda_1": 400.0,
        "lambda_2": 600.0,
        "luminosity_distance": 100.0,
    }
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs={
            "approximant": "IMRPhenomXP_NRTidalv3",
            "f_ref": 20.0,
            # Bilby's convention: a phase shift then also rotates the in-plane
            # spins, which is what makes the (2, 2) dependence exact.
            "spin_conversion_phase": None,
        },
        wfg_domain=domain,
        data_domain=domain,
        event_data={"waveform": waveform, "asds": asds},
        t_ref=1126259462.4,
    )
    assert likelihood.waveform_generator.phase_is_global_factor

    phases = np.array([0.37, 2.9, 5.81])
    terms = likelihood.phase_grid_terms_22(theta)
    from_terms = likelihood.log_likelihood_22_from_terms(terms, phases)
    direct = [likelihood.log_likelihood({**theta, "phase": p}) for p in phases]
    np.testing.assert_allclose(from_terms, direct, rtol=1e-9)

    # Stacked along a leading sample axis, as the synthetic phase evaluates them.
    stacked = {k: np.array([v, v]) for k, v in terms.items()}
    at_drawn = likelihood.log_likelihood_22_from_terms(stacked, phases[:2, None])
    assert at_drawn.shape == (2, 1)
    np.testing.assert_allclose(at_drawn[:, 0], direct[:2], rtol=1e-9)


def test_phase_grid_raise_points_at_the_exact_22_path():
    """With a spin convention the mode sum cannot use, the error says so -- and for a
    model whose phase shift is a global factor it also says that the (2, 2) path is
    exact there, which is the cheaper way out rather than fixing the convention."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=1.0)
    event_data = {
        "waveform": {ifo: np.zeros(len(domain), dtype=complex) for ifo in ["H1"]},
        "asds": {ifo: np.ones(len(domain)) for ifo in ["H1"]},
    }
    for approximant, hinted in (("IMRPhenomXP", True), ("IMRPhenomXPHM", False)):
        likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs={
                "approximant": approximant,
                "f_ref": 20.0,
                "spin_conversion_phase": None,
            },
            wfg_domain=domain,
            data_domain=domain,
            event_data=event_data,
            t_ref=1126259462.4,
        )
        with pytest.raises(ValueError, match="spin_conversion_phase = 0") as excinfo:
            likelihood.phase_grid_terms(THETA)
        assert ("approximation_22_mode" in str(excinfo.value)) is hinted
