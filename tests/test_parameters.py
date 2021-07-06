from dingo.gw.parameters import GWPriorDict, Uniform, Constraint,\
    Sine, Cosine,UniformSourceFrame, Planck15
import pytest
import numpy as np


@pytest.fixture
def parameter_lists():
    params_intrinsic = ['mass_1', 'mass_2', 'mass_ratio',
                       'chirp_mass', 'phase', 'theta_jn',
                       'a_1', 'a_2', 'tilt_1', 'tilt_2',
                       'phi_12', 'phi_jl']
    params_extrinsic = ['luminosity_distance', 'dec', 'ra',
                       'psi', 'geocent_time']
    return params_intrinsic, params_extrinsic


def test_sample_intrinsic(parameter_lists):
    """Check samples drawn from intrinsic prior distribution."""
    priors = GWPriorDict()
    params_intrinsic, _ = parameter_lists
    params_expected = params_intrinsic
    len_expected = len(params_expected)
    size = 42

    assert set(priors.intrinsic_parameters) == set(params_expected)
    # Draw samples from the prior
    sample_dict = priors.sample_intrinsic(size=size)
    assert np.array(list(sample_dict.values())).shape == (len_expected, size)


def test_sample_extrinsic(parameter_lists):
    """Check samples drawn from extrinsic prior distribution."""
    priors = GWPriorDict()
    _, params_extrinsic = parameter_lists
    params_expected = params_extrinsic
    len_expected = len(params_expected)
    size = 67

    assert set(priors.extrinsic_parameters) == set(params_expected)
    # Draw samples from the prior
    sample_dict = priors.sample_extrinsic(size=size)
    assert np.array(list(sample_dict.values())).shape == (len_expected, size)


def test_custom_prior(parameter_lists):
    """Specify a custom prior distribution."""
    params_intrinsic, params_extrinsic = parameter_lists
    prior_dict = {'mass_1': Uniform(minimum=10.0, maximum=100.0, name='mass_1'),
         'mass_2': Uniform(minimum=10.0, maximum=100.0, name='mass_2')}
    priors = GWPriorDict(prior_dict)

    # Intrinsic parameters will include the component masses we specified
    # and phase and theta_jn. Alternatively, these are all intrinsic parameters
    # minus spins and minus mass-ratio and chirp-mass:
    params_expected = set(params_intrinsic)
    params_expected -= {'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl'}
    params_expected -= {'mass_ratio', 'chirp_mass'}
    len_expected = len(params_expected)
    size = 42

    # Draw samples from the intrinsic prior
    sample_dict = priors.sample_intrinsic(size=size)
    assert np.array(list(sample_dict.values())).shape == (len_expected, size)
    assert set(sample_dict.keys()) == set(params_expected)

    # Draw samples from the extrinsic prior
    params_expected = params_extrinsic
    len_expected = len(params_expected)
    sample_dict = priors.sample_extrinsic(size=size)
    assert np.array(list(sample_dict.values())).shape == (len_expected, size)
    assert set(sample_dict.keys()) == set(params_expected)

