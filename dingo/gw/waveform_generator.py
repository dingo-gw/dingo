

class WaveformGenerator(object):

    def __init__(self, approximant, domain):

        self.approximant = approximant
        self.domain = domain

    def generate_hplus_hcross(self, parameters):

        parameters_lal = self.convert_parameters(parameters)



class RandomProjectToDetectors(object):

    def __init__(self, domain, extrinsic_prior):

        self.domain = domain
        self.extrinsic_prior = extrinsic_prior

    def __call__(self, sample):

        extrinsic_parameters = self.extrinsic_prior.sample()
        return self.project_to_detectors(sample['hplus'], sample['hcross'], sample['parameters'], extrinsic_parameters)

    def project_to_detectors(self, hplus, hcross, old_parameters, new_extrinsic_parameters):

        pass