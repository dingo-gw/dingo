import numpy as np
import pytest
from copy import deepcopy

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.likelihood import StationaryGaussianGWLikelihood


@pytest.fixture
def ufd_setup():
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)

    # Set waveform below f_min to 0. (This is important for the test to work!)
    waveform = {
        "H1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "L1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "V1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
    }
    # Set asds below f_min to 1.
    asds = {
        "H1": np.where(domain.frequency_mask, 1e-20, 1.0),
        "L1": np.where(domain.frequency_mask, 1e-20, 1.0),
        "V1": np.where(domain.frequency_mask, 1e-20, 1.0),
    }

    sample = {"waveform": waveform, "asds": asds}

    likelihood_kwargs = {
        "wfg_kwargs": {
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        "wfg_domain": domain,
        "data_domain": domain,
        "event_data": sample,
        "t_ref": 1248242632.0,
    }
    # Create data with smaller range, e.g. [20., 512.]
    f_max_new = 512.0
    domain_small = UniformFrequencyDomain(
        f_min, f_max_new, delta_f=1 / T
    )

    waveform_small = {
        "H1": np.where(domain_small.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "L1": np.where(domain_small.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "V1": np.where(domain_small.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
    }
    asds_small = {
        "H1": np.where(domain_small.frequency_mask, 1e-20, 1.0),
        "L1": np.where(domain_small.frequency_mask, 1e-20, 1.0),
        "V1": np.where(domain_small.frequency_mask, 1e-20, 1.0),
    }

    sample_small = {"waveform": waveform_small, "asds": asds_small}
    likelihood_kwargs_small = {
        "wfg_kwargs": {
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        "wfg_domain": domain_small,
        "data_domain": domain_small,
        "event_data": sample_small,
        "t_ref": 1248242632.0,
    }

    likelihood_kwargs_masked = deepcopy(likelihood_kwargs)
    likelihood_kwargs_masked["frequency_update"] = {
        "maximum_frequency": f_max_new,
    }

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

    return likelihood_kwargs, likelihood_kwargs_small, likelihood_kwargs_masked, theta


@pytest.fixture
def mfd_setup():
    nodes = [20.0, 34.0, 46.0, 62.0, 78.0, 1038.0]
    f_min = 20.0
    f_max = 1038.0
    T = 8.0
    base_domain = UniformFrequencyDomain(
        f_min=f_min, f_max=f_max, delta_f=1 / T
    )
    domain = MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=1 / T, base_domain=base_domain
    )

    # Set waveform below f_min to 0. (This is important for the test to work!)
    waveform = {
        "H1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "L1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "V1": np.where(domain.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
    }
    # Set asds below f_min to 1.
    asds = {
        "H1": np.where(domain.frequency_mask, 1e-20, 1.0),
        "L1": np.where(domain.frequency_mask, 1e-20, 1.0),
        "V1": np.where(domain.frequency_mask, 1e-20, 1.0),
    }

    sample = {"waveform": waveform, "asds": asds}

    likelihood_kwargs = {
        "wfg_kwargs": {
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        "wfg_domain": domain,
        "data_domain": domain,
        "event_data": sample,
        "t_ref": 1248242632.0,
    }
    # Create data with smaller range, e.g. [20., 512.]
    f_max_new = 512.0
    nodes_small = [20.0, 34.0, 46.0, 62.0, 78.0, f_max_new]
    base_domain_small = UniformFrequencyDomain(
        f_min, f_max_new, delta_f=1 / T
    )
    domain_small = MultibandedFrequencyDomain(
        nodes=nodes_small, delta_f_initial=1 / T, base_domain=base_domain_small
    )

    waveform_small = {
        "H1": np.where(domain_small.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "L1": np.where(domain_small.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
        "V1": np.where(domain_small.frequency_mask, (1.0 + 1j) * 1e-20, 0.0),
    }
    asds_small = {
        "H1": np.where(domain_small.frequency_mask, 1e-20, 1.0),
        "L1": np.where(domain_small.frequency_mask, 1e-20, 1.0),
        "V1": np.where(domain_small.frequency_mask, 1e-20, 1.0),
    }

    sample_small = {"waveform": waveform_small, "asds": asds_small}
    likelihood_kwargs_small = {
        "wfg_kwargs": {
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
        "wfg_domain": domain_small,
        "data_domain": domain_small,
        "event_data": sample_small,
        "t_ref": 1248242632.0,
    }

    likelihood_kwargs_masked = deepcopy(likelihood_kwargs)
    likelihood_kwargs_masked["frequency_update"] = {
        "maximum_frequency": f_max_new,
    }

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

    return likelihood_kwargs, likelihood_kwargs_small, likelihood_kwargs_masked, theta


@pytest.mark.parametrize(
    "setup",
    [
        "ufd_setup",
        "mfd_setup",
    ],
)
def test_likelihood_frequency_masking(request, setup):
    likelihood_kwargs, likelihood_kwargs_small, likelihood_kwargs_masked, theta = (
        request.getfixturevalue(setup)
    )

    likelihood = StationaryGaussianGWLikelihood(
        **likelihood_kwargs
    )  # domain: [20, 1024]
    likelihood_small_range = StationaryGaussianGWLikelihood(
        **likelihood_kwargs_small
    )  # domain: [20, 512]
    likelihood_masked = StationaryGaussianGWLikelihood(
        **likelihood_kwargs_masked
    )  # domain: [20, 512]

    l_vals = likelihood.log_likelihood(theta)
    l_vals_small_range = likelihood_small_range.log_likelihood(theta)
    l_vals_masked = likelihood_masked.log_likelihood(theta)

    assert not np.isclose(l_vals, l_vals_small_range)
    assert not np.isclose(l_vals, l_vals_masked)
    assert np.isclose(l_vals_small_range, l_vals_masked)
