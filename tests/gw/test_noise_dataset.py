import pytest
import h5py
import numpy as np

from dingo.gw.domains import build_domain
from dingo.gw.noise_dataset import ASDDataset

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



def test_noise_dataset_load(noise_dataset):
    dataset_path, domain_dict, ifos_num_asds = noise_dataset

    asd_dataset = ASDDataset(dataset_path)
    # check domain
    assert asd_dataset.domain.domain_dict == domain_dict
    # check that the dataset has the correct ifos and corresponding datasets
    assert asd_dataset.asds.keys() == ifos_num_asds.keys()
    for ifo, n in ifos_num_asds.items():
        assert asd_dataset.asds[ifo].shape == (n, len(asd_dataset.domain))
    # check that sampling work correctly
    asd_samples = asd_dataset.sample_random_asds()
    assert asd_samples.keys() == ifos_num_asds.keys()
    for _, asd in asd_samples.items():
        assert len(asd) == len(asd_dataset.domain)

    # repeat tests for ASDDataset with only a subset of detectors
    ifos = ['H1', 'V1']
    asd_dataset = ASDDataset(dataset_path, ifos)
    # check domain
    assert asd_dataset.domain.domain_dict == domain_dict
    # check that the dataset has the correct ifos and corresponding datasets
    assert set(asd_dataset.asds.keys()) == set(ifos)
    for ifo in ifos:
        assert asd_dataset.asds[ifo].shape == (ifos_num_asds[ifo],
                                               len(asd_dataset.domain))
    # check that sampling work correctly
    asd_samples = asd_dataset.sample_random_asds()
    assert set(asd_samples.keys()) == set(ifos)
    for _, asd in asd_samples.items():
        assert len(asd) == len(asd_dataset.domain)


def test_noise_dataset_truncate_domain(noise_dataset):
    dataset_path, domain_dict, ifos_num_asds = noise_dataset

    asd_dataset = ASDDataset(dataset_path)
    asd_samples = asd_dataset.sample_random_asds()

    new_range = (20,40)
    asd_dataset_truncated = ASDDataset(dataset_path)
    asd_dataset_truncated.truncate_dataset_domain(new_range=new_range)
    asd_samples_truncated = asd_dataset_truncated.sample_random_asds()

    domain_ref = build_domain(domain_dict)
    domain_ref.set_new_range(*new_range)
    nf_tr = domain_ref.len_truncated
    nf = len(asd_dataset.domain)

    for ifo, n in ifos_num_asds.items():
        asd_data_full = asd_dataset.asds[ifo]
        asd_data_full_truncated = asd_dataset_truncated.asds[ifo]
        print('cp')
        # check that asd data and asd samples have the correct dimensions
        assert asd_data_full.shape == (n, nf)
        assert len(asd_samples[ifo]) == nf
        assert asd_data_full_truncated.shape == (n, nf_tr)
        assert len(asd_samples_truncated[ifo]) == nf_tr
        # check that the truncation worked properly
        assert np.all(domain_ref.truncate_data(
            asd_data_full, allow_for_flexible_upper_bound=True) \
               == asd_data_full_truncated)