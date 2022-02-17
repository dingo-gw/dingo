import numpy as np
from dingo.gw.prior import BBHExtrinsicPriorDict


class SampleExtrinsicParameters(object):
    """
    Sample extrinsic parameters and add them to sample in a separate dictionary.
    """
    def __init__(self, extrinsic_prior_dict):
        self.extrinsic_prior_dict = extrinsic_prior_dict
        self.prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = self.prior.sample()
        sample['extrinsic_parameters'] = extrinsic_parameters
        return sample

    @property
    def reproduction_dict(self):
        return {'extrinsic_prior_dict': self.extrinsic_prior_dict}


class SelectStandardizeRepackageParameters(object):
    """
    This transformation selects the parameters in standardization_dict,
    normalizes them by setting p = (p - mean) / std, and repackages the
    selected parameters to a numpy array.
    """
    def __init__(self, standardization_dict):
        self.mean = standardization_dict['mean']
        self.std = standardization_dict['std']
        self.N = len(self.mean.keys())
        self.regression_parameters = list(self.mean.keys())
        if self.mean.keys() != self.std.keys():
            raise ValueError('Keys of means and stds do not match.')

    def __call__(self, input_sample):
        sample = input_sample.copy()
        parameters = np.empty(self.N, dtype=np.float32)
        for idx, par in enumerate(self.regression_parameters):
            parameters[idx] = \
                (sample['parameters'][par] - self.mean[par]) / self.std[par]
        sample['parameters'] = parameters
        return sample


class StandardizeParameters:
    """
    Standardize parameters according to the transform (x - mu) / std.
    """
    def __init__(self, mu, std):
        """
        Initialize the standardization transform with means
        and standard deviations for each parameter

        Parameters
        ----------
        mu : Dict[str, float]
            The (estimated) means
        std : Dict[str, float]
            The (estimated) standard deviations
        """
        self.mu = mu
        self.std = std
        if not set(mu.keys()) == set(std.keys()):
            raise ValueError('The keys in mu and std disagree:'
                             f'mu: {mu.keys()}, std: {std.keys()}')

    def __call__(self, samples):
        """Standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform',
            'noise_summary'.

        Only parameters included in mu, std get transformed.
        """
        x = samples['parameters']
        y = {k: (x[k] - self.mu[k]) / self.std[k] for k in self.mu.keys()}
        samples_out = samples.copy()
        samples_out['parameters'] = y
        return samples_out

    def inverse(self, samples):
        """De-standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform',
            'noise_summary'.

        Only parameters included in mu, std get transformed.
        """
        y = samples['parameters']
        x = {k: self.mu[k] + y[k] * self.std[k] for k in self.mu.keys()}
        samples_out = samples.copy()
        samples_out['parameters'] = x
        return samples_out