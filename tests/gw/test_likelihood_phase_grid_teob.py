"""Phase-grid likelihood for TEOBResumSDALI.

TEOB modes carry a phase-dependent time shift (the epoch is set by the peak of h+,
following gwsignal), bookkept in wfg._deferred_timeshift_data. The phase-grid
likelihood must (a) agree with the direct likelihood at any phase, and (b) agree
with an explicit per-phase resum of the modes, while (c) not recomputing the epoch
one phase at a time."""

import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.injection import Injection
from dingo.gw.likelihood import StationaryGaussianGWLikelihood, inner_product
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import wfg_utils
from dingo.gw.waveform_generator.waveform_generator import sum_contributions_m

pytest.importorskip("EOBRun_module")

T_REF = 1126259462.391
WFG_KWARGS = {
    "approximant": "TEOBResumSDALI",
    "f_ref": 10.0,
    "f_start": 10.0,
    "spin_conversion_phase": 0.0,
    "new_interface": True,
}
THETA_INJ = {
    "chirp_mass": 40.0,
    "mass_ratio": 0.8,
    "chi_1": 0.2,
    "chi_2": -0.1,
    "eccentricity": 0.15,
    "mean_per_ano": 1.0,
    "theta_jn": 0.9,
    "luminosity_distance": 300.0,
    "geocent_time": 0.01,
    "ra": 1.2,
    "dec": -0.4,
    "psi": 0.7,
    "phase": 0.0,
}


@pytest.fixture(scope="module")
def teob_likelihood():
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)
    ifos = ["H1", "L1", "V1"]
    prior = build_prior_with_defaults(
        {
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=60.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chi_1": 'bilby.gw.prior.AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.9))',
            "chi_2": 'bilby.gw.prior.AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.9))',
            "eccentricity": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.3)",
            "mean_per_ano": "bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi)",
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=5000.0)",
            "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "ra": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)",
            "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic")',
        }
    )
    injection = Injection(
        prior=prior,
        wfg_kwargs=WFG_KWARGS,
        wfg_domain=domain,
        data_domain=domain,
        ifo_list=ifos,
        t_ref=T_REF,
    )
    injection.asd = {ifo: np.full(len(domain), 1e-22) for ifo in ifos}
    np.random.seed(42)
    event_data = injection.injection(THETA_INJ)

    return StationaryGaussianGWLikelihood(
        wfg_kwargs=WFG_KWARGS,
        wfg_domain=domain,
        data_domain=domain,
        event_data=event_data,
        t_ref=T_REF,
    )


@pytest.fixture(scope="module")
def theta():
    # Slightly off the injection so the likelihood has non-trivial structure.
    return {**THETA_INJ, "chirp_mass": 40.05, "mean_per_ano": 1.1}


def test_phase_grid_matches_explicit_per_phase_resum(teob_likelihood, theta):
    likelihood = teob_likelihood
    phases = np.linspace(0, 2 * np.pi, 257)

    grid = likelihood.log_likelihood_phase_grid(theta, phases)

    pol_m = likelihood.signal_m({**theta, "phase": 0.0})
    pol_m = {m: pol["waveform"] for m, pol in pol_m.items()}
    dts = likelihood.waveform_generator._deferred_timeshift_data
    assert dts is not None, "TEOB should set a deferred time shift"
    d = likelihood.whitened_strains

    reference = np.empty(len(phases))
    for idx, phi in enumerate(phases):
        mu = sum_contributions_m(pol_m, phase_shift=phi, deferred_timeshift_data=dts)
        rho2opt = sum(inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values())
        kappa2 = sum(inner_product(d[ifo], mu[ifo]) for ifo in mu)
        reference[idx] = likelihood.log_Zn + kappa2 - 0.5 * rho2opt

    # The logL varies by O(100) over the grid; agreement should be at rounding level.
    assert np.ptp(reference) > 10
    np.testing.assert_allclose(grid, reference, rtol=0, atol=1e-6 * np.ptp(reference))


def test_phase_grid_matches_direct_likelihood(teob_likelihood, theta):
    likelihood = teob_likelihood
    phases = np.linspace(0, 2 * np.pi, 7)[:-1]

    grid = likelihood.log_likelihood_phase_grid(theta, phases)
    direct = np.array(
        [likelihood.log_likelihood({**theta, "phase": phi}) for phi in phases]
    )

    # The mode path tapers/FFTs individual modes while the direct path conditions the
    # polarizations, so differences of O(0.1) nats are expected at this SNR (the
    # likelihood spans ~60 nats over the phase circle). A wrong time shift would be
    # off by many nats.
    np.testing.assert_allclose(grid, direct, rtol=0, atol=0.3)


def test_phase_grid_does_not_recompute_epoch_per_phase(
    teob_likelihood, theta, monkeypatch
):
    """Generating the TEOB modes legitimately computes the epoch once (at the
    reference phase). The phase grid must not call the scalar routine once more per
    grid phase on top of that."""
    calls = []
    original = wfg_utils.compute_epoch_from_resized_td_modes

    def _counting(*args, **kwargs):
        calls.append(args[2])  # the phase argument
        return original(*args, **kwargs)

    monkeypatch.setattr(wfg_utils, "compute_epoch_from_resized_td_modes", _counting)

    phases = np.linspace(0, 2 * np.pi, 101)
    grid = teob_likelihood.log_likelihood_phase_grid(theta, phases)

    assert np.all(np.isfinite(grid))
    assert len(calls) == 1, f"scalar epoch routine called {len(calls)} times"
