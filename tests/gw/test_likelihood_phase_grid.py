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
