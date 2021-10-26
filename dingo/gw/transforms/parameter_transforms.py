from dingo.gw.prior_split import BBHExtrinsicPriorDict


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