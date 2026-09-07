"""
The calibration-marginalized likelihood is evaluated in matrix-vector form with the
calibration curves attached to the signal. This checks it against the original
formulation, which expands the waveform into (num_curves, N) and evaluates the inner
products per curve.
"""

import os

import numpy as np
import pandas as pd
from bilby.core.utils import random as bilby_random

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.likelihood import StationaryGaussianGWLikelihood

ENVELOPE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "transforms",
    "calibration_envelope_test.txt",
)
IFOS = ["H1", "L1", "V1"]


def _event_data(domain, seed=0):
    rng = np.random.default_rng(seed)
    n = len(domain)
    with np.errstate(divide="ignore"):
        asd_shape = 1e-23 * np.sqrt(1 + (domain() / 100.0) ** -4)
    asds = {ifo: np.where(domain.frequency_mask, asd_shape, 1.0) for ifo in IFOS}
    waveform = {}
    for ifo in IFOS:
        strain = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * asds[ifo]
        strain *= domain.noise_std
        strain[~domain.frequency_mask] = 0.0
        waveform[ifo] = strain
    return {"waveform": waveform, "asds": asds}


def _theta(n, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "chirp_mass": rng.uniform(25.0, 60.0, n),
            "mass_ratio": rng.uniform(0.3, 1.0, n),
            "a_1": rng.uniform(0.0, 0.9, n),
            "a_2": rng.uniform(0.0, 0.9, n),
            "tilt_1": rng.uniform(0.0, np.pi, n),
            "tilt_2": rng.uniform(0.0, np.pi, n),
            "phi_12": rng.uniform(0.0, 2 * np.pi, n),
            "phi_jl": rng.uniform(0.0, 2 * np.pi, n),
            "theta_jn": rng.uniform(0.0, np.pi, n),
            "luminosity_distance": rng.uniform(500.0, 3000.0, n),
            "geocent_time": rng.uniform(-0.05, 0.05, n),
            "dec": rng.uniform(-1.4, 1.4, n),
            "ra": rng.uniform(0.0, 2 * np.pi, n),
            "psi": rng.uniform(0.0, np.pi, n),
            "phase": rng.uniform(0.0, 2 * np.pi, n),
        }
    )


def test_calibration_marginalized_likelihood_matches_expanded_form():
    domain = UniformFrequencyDomain(20.0, 512.0, delta_f=0.25)
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs={
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        wfg_domain=domain,
        data_domain=domain,
        event_data=_event_data(domain),
        t_ref=1248242632.0,
        calibration_marginalization_kwargs={
            "calibration_envelope": {ifo: ENVELOPE for ifo in IFOS},
            "num_calibration_nodes": 10,
            "num_calibration_curves": 20,
        },
    )
    theta = _theta(3)
    for i, row in enumerate(theta.to_dict("records")):
        # Calibration draws are random; seed bilby's generator so that the likelihood
        # and the reference computation below see the same curves.
        bilby_random.seed(100 + i)
        ll = likelihood.log_likelihood(row)

        bilby_random.seed(100 + i)
        signal = likelihood.signal(row)
        d = likelihood.whitened_strains
        rho2opt = 0.0
        kappa2 = 0.0
        for ifo in IFOS:
            mu_expanded = signal["waveform"][ifo] * signal["calibration_curves"][ifo]
            assert mu_expanded.shape == (20, len(domain))
            rho2opt = rho2opt + np.sum(np.abs(mu_expanded) ** 2, axis=1)
            kappa2 = kappa2 + np.sum(d[ifo].conj() * mu_expanded, axis=1).real
        per_curve = likelihood.log_Zn + kappa2 - 0.5 * rho2opt
        ref = np.logaddexp.reduce(per_curve) - np.log(len(per_curve))
        assert np.isclose(ll, ref, rtol=0.0, atol=1e-6)
