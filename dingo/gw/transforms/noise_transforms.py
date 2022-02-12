import numpy as np

class SampleNoiseASD(object):
    """
    Sample a random asds for each detector and add them to sample['asds'].
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

class AddWhiteNoiseComplex(object):
    """
    Adds white noise with a standard deviation determined by self.scale to the
    complex strain data.
    """
    def __init__(self, scale = 1.0):
        self.scale = scale

    def __call__(self, input_sample):
        sample = input_sample.copy()
        noisy_strains = {}
        for ifo, pure_strain in sample['waveform'].items():
            noise = \
                (np.random.normal(scale=self.scale, size=len(pure_strain))
                + np.random.normal(scale=self.scale, size=len(pure_strain)) *1j)
            noise = noise.astype(np.complex64)
            noisy_strains[ifo] = pure_strain + noise
        sample['waveform'] = noisy_strains
        return sample


class RepackageStrainsAndASDS(object):
    """
    Repackage the strains and the asds into an [num_ifos, 3, num_bins]
    dimensional tensor. Order of ifos is provided by self.ifos. By
    convention, [:,i,:] is used for:
        i = 0: strain.real
        i = 1: strain.imag
        i = 2: 1 / (asd * 1e23)
    """
    def __init__(self, ifos, first_index=0):
        self.ifos = ifos
        self.first_index = first_index

    def __call__(self, input_sample):
        sample = input_sample.copy()
        strains = np.empty((len(self.ifos),3,len(sample['asds'][self.ifos[0]])-self.first_index),
                           dtype=np.float32)
        for idx_ifo, ifo in enumerate(self.ifos):
            strains[idx_ifo,0] = sample['waveform'][ifo][self.first_index:].real
            strains[idx_ifo,1] = sample['waveform'][ifo][self.first_index:].imag
            strains[idx_ifo,2] = 1 / (sample['asds'][ifo][self.first_index:] * 1e23)
        sample['waveform'] = strains
        return sample

if __name__ == '__main__':
    AD = ASDDataset('../../../data/PSDs/asds_O1.hdf5')
    asd_samples = AD.sample_random_asds()