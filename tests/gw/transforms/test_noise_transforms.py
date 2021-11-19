import pytest
import h5py
import numpy as np

from dingo.gw.domains import build_domain
from dingo.gw.noise_dataset import ASDDataset
from dingo.gw.transforms import SampleNoiseASD, WhitenStrain, \
    WhitenAndScaleStrain, AddWhiteNoiseComplex

@pytest.fixture
def noise_dataset():
    dataset_path = './asds_toydataset.hdf5'

    domain_dict = {'name': 'UniformFrequencyDomain',
                   'kwargs': {'f_min': 0.0, 'f_max': 100.0, 'delta_f': 1.0}}
    domain = build_domain(domain_dict)
    domain_dict = domain.domain_dict
    ifos_num_asds = {'H1': 10, 'L1': 8, 'V1': 5}

    f = h5py.File(dataset_path, 'w')
    gps_times = f.create_group('gps_times')
    settings = {'domain_dict': domain.domain_dict}

    for ifo, num_asds in ifos_num_asds.items():
        asds = np.sin(np.outer(np.arange(num_asds)/100, domain()))
        f.create_dataset(f'asds_{ifo}', data=asds)
        gps_times[ifo] = np.arange(num_asds)

    f.attrs['metadata'] = str(settings)
    f.close()

    return dataset_path, domain_dict, ifos_num_asds

@pytest.fixture
def input_sample():
    num_bins = 100
    ifos = ['H1', 'L1', 'V1']
    input_sample = {'waveform': {}, 'asds': {}}
    for idx, ifo in enumerate(ifos):
        input_sample['waveform'][ifo] = np.ones(num_bins) * (idx + 1)
        input_sample['asds'][ifo] = np.random.rand(num_bins)
    return input_sample

def test_sample_random_asds(noise_dataset):
    dataset_path, domain_dict, ifos_num_asds = noise_dataset
    asd_dataset = ASDDataset(dataset_path)
    sample_noise_asd = SampleNoiseASD(asd_dataset)
    sample = sample_noise_asd({})
    assert sample['asds'].keys() == asd_dataset.asds.keys()
    for k, v in sample['asds'].items():
        assert len(v) == len(asd_dataset.domain)
        # check that asd is indeed sampled from one of the asds in the dataset
        assert np.sum(np.all(v == asd_dataset.asds[k], axis=1)) == 1

def test_whiten_strain(input_sample):
    sample_in = input_sample
    ifos = sample_in['waveform'].keys()
    whiten_strain = WhitenStrain()
    sample_out = whiten_strain(sample_in)
    assert sample_in['asds'] == sample_out['asds']
    # check that whitening works as intended
    for ifo in ifos:
        assert np.allclose(sample_out['waveform'][ifo] * sample_out['asds'][
            ifo],sample_in['waveform'][ifo])
    # check that ValueError is raised, if waveform and asds have different ifos
    sample_in['waveform'].pop(list(ifos)[0])
    with pytest.raises(ValueError):
        whiten_strain(sample_in)

def test_whiten_and_scale_strain(input_sample):
    scale_factor = 2
    sample_in = input_sample
    ifos = sample_in['waveform'].keys()
    whiten_and_scale_strain = WhitenAndScaleStrain(scale_factor=scale_factor)
    sample_out = whiten_and_scale_strain(sample_in)
    assert sample_in['asds'] == sample_out['asds']
    # check that whitening and scaling works as intended
    for ifo in ifos:
        assert np.allclose(sample_out['waveform'][ifo] * sample_out['asds'][
            ifo],sample_in['waveform'][ifo]/scale_factor)
    # check that ValueError is raised if waveform and asds have different ifos
    sample_in['waveform'].pop(list(ifos)[0])
    with pytest.raises(ValueError):
        whiten_and_scale_strain(sample_in)
    # check that ValueError is raised if no scale_factor is provided
    with pytest.raises(TypeError):
        _ = WhitenAndScaleStrain()

def test_add_white_noise_complex(input_sample):
    tol = 0.15
    noise_scale = 2
    sample_in = input_sample
    ifos = sample_in['waveform'].keys()
    add_white_noise_complex = AddWhiteNoiseComplex(scale=noise_scale)
    sample_out = add_white_noise_complex(sample_in)
    # check that noise with correct std is added
    for ifo in ifos:
        std_real = np.std(sample_out['waveform'][ifo].real)
        std_imag = np.std(sample_out['waveform'][ifo].imag)
        mean_real = np.mean(sample_out['waveform'][ifo].real)
        mean_imag = np.mean(sample_out['waveform'][ifo].imag)
        assert std_real != std_imag
        assert abs(1 - std_real / noise_scale) < tol
        assert abs(1 - std_imag / noise_scale) < tol
        assert abs(np.mean(sample_in['waveform'][ifo].real) - mean_real) < tol
        assert abs(np.mean(sample_in['waveform'][ifo].imag) - mean_imag) < tol