import numpy as np

class SampleNoiseASD(object):
    """
    TODO
    """
    def __init__(self, asd_dataset):
        self.asd_dataset = asd_dataset

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample['asds'] = self.asd_dataset.sample_random_asds()
        return sample

class WhitenStrain(object):
    """
    Whiten the strain data by dividing w.r.t. the corresponding asds.
    """
    def __init__(self):
        pass

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ifos = sample['waveform'].keys()
        if ifos != sample['asds'].keys():
            raise ValueError(f'Detectors of strain data, {ifos}, do not match '
                             f'those of asds, {sample["asds"].keys()}.')
        whitened_strains = \
            {ifo: sample['waveform'][ifo] / sample['asds'][ifo] for ifo in ifos}
        sample['waveform'] = whitened_strains
        return sample

class WhitenAndScaleStrain(object):
    """
    Whiten the strain data by dividing w.r.t. the corresponding asds,
    and scale it with 1/scale_factor.

    In uniform frequency domain the scale factor should be
    np.sqrt(window_factor) / np.sqrt(4.0 * delta_f).
    It has two purposes:
        (*) the denominator accounts for frequency binning
        (*) dividing by window factor accounts for windowing of strain data
    """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ifos = sample['waveform'].keys()
        if ifos != sample['asds'].keys():
            raise ValueError(f'Detectors of strain data, {ifos}, do not match '
                             f'those of asds, {sample["asds"].keys()}.')
        whitened_strains = \
            {ifo: sample['waveform'][ifo] /
                  (sample['asds'][ifo] * self.scale_factor) for ifo in ifos}
        sample['waveform'] = whitened_strains
        return sample

class TruncateStrainAndASD(object):
    """
    Truncate the strain and asd data to the specified range. This corresponds
    to truncating the likelihood integral accordingly.
    """
    def __init__(self, strain_domain, asd_domain, truncation_range):
        """
        :param strain_domain: domain object of the strain data
        :param asd_domain: domain object of the asd data
        :param truncation_range: frequency range for truncation
        """
        self.strain_domain = strain_domain
        self.asd_domain = asd_domain
        # initialize truncations
        self.strain_domain.initialize_truncation(truncation_range)
        self.asd_domain.initialize_truncation(truncation_range)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ifos = sample['waveform'].keys()
        if ifos != sample['asds'].keys():
            raise ValueError(f'Detectors of strain data, {ifos}, do not match '
                             f'those of asds, {sample["asds"].keys()}.')
        truncated_strains =  \
            {ifo: self.strain_domain.truncate_data(sample['waveform'][ifo])
             for ifo in ifos}
        truncated_asds =  \
            {ifo: self.asd_domain.truncate_data(sample['asds'][ifo])
             for ifo in ifos}
        sample['waveform'] = truncated_strains
        sample['asds'] = truncated_asds
        return sample


if __name__ == '__main__':
    AD = ASDDataset('../../../data/PSDs/asds_O1.hdf5')
    asd_samples = AD.sample_random_asds()