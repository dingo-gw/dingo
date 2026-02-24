import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.likelihood import (
    StationaryGaussianGWLikelihood,
    inner_product,
    inner_product_complex,
)


@pytest.fixture
def likelihood_and_theta():
    """Set up a simple likelihood for testing."""
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)

    waveform = {
        "H1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "L1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
    }
    asds = {
        "H1": np.where(domain.frequency_mask, 1e-20, 1.0),
        "L1": np.where(domain.frequency_mask, 1e-20, 1.0),
    }

    sample = {"waveform": waveform, "asds": asds}

    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs={
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        wfg_domain=domain,
        data_domain=domain,
        event_data=sample,
        t_ref=1248242632.0,
    )

    theta = {
        "chirp_mass": 50.0,
        "mass_ratio": 0.8,
        "a_1": 0.3,
        "a_2": 0.4,
        "tilt_1": 1.3,
        "tilt_2": 1.4,
        "phi_12": 0.3,
        "phi_jl": 3.3,
        "theta_jn": 1.2,
        "luminosity_distance": 4000.0,
        "geocent_time": -0.03,
        "dec": -0.3,
        "ra": 1.5,
        "psi": 1.2,
        "phase": 1.4,
    }

    return likelihood, theta


class TestLikelihoodDecomposition:
    """Test the likelihood decomposition log L = log_Zn + kappa2 - 0.5*rho2opt."""

    def test_log_likelihood_decomposition(self, likelihood_and_theta):
        """log L = log_Zn + Re(d|h) - 0.5*(h|h)"""
        likelihood, theta = likelihood_and_theta
        log_l = likelihood.log_likelihood(theta)
        d_inner_h, rho2opt = likelihood._d_inner_h_complex(theta)
        kappa2 = d_inner_h.real
        reconstructed = likelihood.log_Zn + kappa2 - 0.5 * rho2opt
        assert np.isclose(log_l, reconstructed)

    def test_rho2opt_is_positive(self, likelihood_and_theta):
        """rho2opt = (h|h) should be non-negative."""
        likelihood, theta = likelihood_and_theta
        _, rho2opt = likelihood._d_inner_h_complex(theta)
        assert rho2opt >= 0

    def test_rho2opt_matches_direct_computation(self, likelihood_and_theta):
        """rho2opt from _d_inner_h_complex should match direct (h|h)."""
        likelihood, theta = likelihood_and_theta
        _, rho2opt = likelihood._d_inner_h_complex(theta)

        mu = likelihood.signal(theta)["waveform"]
        rho2opt_direct = sum(
            [inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()]
        )
        assert np.isclose(rho2opt, rho2opt_direct)


class TestDInnerHComplex:
    """Test the complex inner product methods."""

    def test_d_inner_h_complex_returns_complex(self, likelihood_and_theta):
        """d_inner_h_complex returns a complex scalar (backward compat)."""
        likelihood, theta = likelihood_and_theta
        result = likelihood.d_inner_h_complex(theta)
        assert np.isscalar(result) or isinstance(
            result, (complex, np.complexfloating)
        )

    def test_private_returns_tuple(self, likelihood_and_theta):
        """_d_inner_h_complex returns (d_inner_h, rho2opt) tuple."""
        likelihood, theta = likelihood_and_theta
        result = likelihood._d_inner_h_complex(theta)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_public_matches_private(self, likelihood_and_theta):
        """d_inner_h_complex matches _d_inner_h_complex[0]."""
        likelihood, theta = likelihood_and_theta
        public = likelihood.d_inner_h_complex(theta)
        private_tuple = likelihood._d_inner_h_complex(theta)
        assert np.isclose(public, private_tuple[0])

    def test_multi_without_rho2opt(self, likelihood_and_theta):
        """d_inner_h_complex_multi without return_rho2opt returns array only."""
        likelihood, theta = likelihood_and_theta
        theta_df = pd.DataFrame([theta, theta])
        result = likelihood.d_inner_h_complex_multi(theta_df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.iscomplexobj(result)

    def test_multi_with_rho2opt(self, likelihood_and_theta):
        """d_inner_h_complex_multi with return_rho2opt returns tuple."""
        likelihood, theta = likelihood_and_theta
        theta_df = pd.DataFrame([theta, theta, theta])
        d_inner_h, rho2opt = likelihood.d_inner_h_complex_multi(
            theta_df, return_rho2opt=True
        )
        assert d_inner_h.shape == (3,)
        assert rho2opt.shape == (3,)
        assert np.iscomplexobj(d_inner_h)
        assert not np.iscomplexobj(rho2opt)
        assert np.all(rho2opt >= 0)

    def test_multi_rho2opt_consistency(self, likelihood_and_theta):
        """rho2opt from multi matches single-sample computation."""
        likelihood, theta = likelihood_and_theta
        d_inner_h_single, rho2opt_single = likelihood._d_inner_h_complex(theta)

        theta_df = pd.DataFrame([theta])
        d_inner_h_multi, rho2opt_multi = likelihood.d_inner_h_complex_multi(
            theta_df, return_rho2opt=True
        )
        assert np.isclose(d_inner_h_single, d_inner_h_multi[0])
        assert np.isclose(rho2opt_single, rho2opt_multi[0])
