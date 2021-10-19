from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import fill_from_fixed_priors, convert_to_lal_binary_black_hole_parameters
from bilby.core.prior import Uniform, Sine, Cosine

import numpy as np
from typing import Dict, Set, Any
import warnings

# Silence INFO and WARNING messages from bilby
import logging

logging.getLogger('bilby').setLevel("ERROR")


class BBHExtrinsicPriorDict(BBHPriorDict):
    """
    This class is the same as BBHPriorDict except that it does not require mass parameters.

    It also contains a method for estimating the standardization parameters.

    TODO:
        * Add support for zenith/azimuth
        * Defaults?
    """

    def default_conversion_function(self, sample):
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)

        # The previous call always adds phi_jl, phi_12 parameters. These are not needed so they can be deleted.
        del out_sample['phi_jl']
        del out_sample['phi_12']

        return out_sample

    def mean_std(self, keys=([]), sample_size=50000):
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys:
            A list of desired parameters
        sample_size:
            For nonanalytic priors, number samples to use to estimate the result.

        Returns dictionaries for the means and standard deviations.

        TODO: Fix for constrained priors. Shouldn't be an issue for extrinsic parameters.
        """
        mean = {}
        std = {}

        # First try to calculate analytically (works for standard priors)
        estimation_keys = []
        for key in keys:
            p = self[key]
            # A few analytic cases
            if isinstance(p, Uniform):
                mean[key] = (p.maximum + p.minimum) / 2.0
                std[key] = np.sqrt((p.maximum - p.minimum)**2.0 / 12.0)
            elif isinstance(p, Sine) and p.minimum == 0.0 and p.maximum == np.pi:
                mean[key] = np.pi / 2.0
                std[key] = np.sqrt(0.25 * (np.pi**2) - 2)
            elif isinstance(p, Cosine) and p.minimum == -np.pi/2 and p.maximum == np.pi/2:
                mean[key] = 0.0
                std[key] = np.sqrt(0.25 * (np.pi**2) - 2)
            else:
                estimation_keys.append(key)

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample_subset(keys, size=sample_size)
            samples = self.default_conversion_function(samples)
            for key in estimation_keys:
                if key in samples.keys():
                    mean[key] = np.mean(samples[key])
                    std[key] = np.std(samples[key])

        return mean, std

# TODO: Add latex labels, names


default_extrinsic_dict = {
    'dec': 'bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)',
    'ra': 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")',
    'geocent_time': 'bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)',
    'psi': 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic")',
    'luminosity_distance': 'bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0)',
}

default_intrinsic_dict = {
    'mass_1': 'bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)',
    'mass_2': 'bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)',
    'mass_ratio': 'bilby.core.prior.Uniform(minimum=0.125, maximum=1.0)',
    'chirp_mass': 'bilby.core.prior.Uniform(minimum=25.0, maximum=100.0)',
    'luminosity_distance': 1000.0,
    'theta_jn': 'bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)',
    'phase': 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
    'a_1': 'bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)',
    'a_2': 'bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)',
    'tilt_1': 'bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)',
    'tilt_2': 'bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)',
    'phi_12': 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
    'phi_jl': 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
    'geocent_time': 0.0,
}
