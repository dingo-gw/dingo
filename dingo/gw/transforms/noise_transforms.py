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
    TODO
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
        # TODO: noise_std
        return sample

if __name__ == '__main__':
    AD = ASDDataset('../../../data/PSDs/asds_O1.hdf5')
    asd_samples = AD.sample_random_asds()