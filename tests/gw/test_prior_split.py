import numpy as np
import pytest
from dingo.gw.prior import BBHExtrinsicPriorDict, BBHPriorDict, default_extrinsic_dict, default_intrinsic_dict


def test_prior_constraint():
    dict = {
        'mass_1': 'bilby.core.prior.Uniform(minimum=10.0, maximum=80.0)',
        'mass_2': 'bilby.core.prior.Uniform(minimum=10.0, maximum=80.0)',
        'mass_ratio': 'bilby.core.prior.Constraint(minimum=0.125, maximum=1.0)',
    }
    prior = BBHPriorDict(dict)
    samples = prior.sample(1000)
    assert np.all(samples['mass_1'] > samples['mass_2'])


def test_mean_std():
    num_samples = 100000
    eps = 0.01
    keys = ['ra', 'dec', 'luminosity_distance']
    prior = BBHExtrinsicPriorDict(default_extrinsic_dict)
    mean_exact, std_exact = prior.mean_std(keys)
    mean_approx, std_approx = prior.mean_std(keys,
                                             sample_size=num_samples,
                                             force_numerical=True)
    ratios_exact = (np.array(list(mean_exact.values())) /
                    np.array(list(std_exact.values())))
    ratios_approx = (np.array(list(mean_approx.values())) /
                     np.array(list(std_approx.values())))
    assert list(mean_exact.keys()) == keys
    assert np.allclose(ratios_exact, ratios_approx,
                       atol=eps, rtol=eps)
