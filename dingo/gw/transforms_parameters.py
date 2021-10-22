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