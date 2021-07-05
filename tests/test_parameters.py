from dingo.gw.parameters import GWPriorDict
import numpy as np


def test_sample_intrinsic():
    """Check samples drawn from intrinsic prior distribution."""
    priors = GWPriorDict()
    params_expected = ['mass_1', 'mass_2', 'mass_ratio',
                       'chirp_mass', 'phase', 'theta_jn',
                       'a_1', 'a_2', 'tilt_1', 'tilt_2',
                       'phi_12', 'phi_jl']
    len_expected = len(params_expected)
    size = 42

    assert set(priors.intrinsic_parameters) == set(params_expected)
    # Draw samples from the prior
    sample_dict = priors.sample_intrinsic(size=size)
    assert np.array(list(sample_dict.values())).shape == (len_expected, size)


def test_sample_extrinsic():
    """Check samples drawn from extrinsic prior distribution."""
    priors = GWPriorDict()
    params_expected = ['luminosity_distance', 'dec', 'ra',
                       'psi', 'geocent_time']
    len_expected = len(params_expected)
    size = 67

    assert set(priors.extrinsic_parameters) == set(params_expected)
    # Draw samples from the prior
    sample_dict = priors.sample_extrinsic(size=size)
    assert np.array(list(sample_dict.values())).shape == (len_expected, size)
