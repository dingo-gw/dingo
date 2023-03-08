import pytest
import h5py
import numpy as np

from dingo.gw.domains import build_domain
from dingo.gw.noise.asd_dataset import ASDDataset


@pytest.fixture
def noise_dataset():
    dataset_path = "./asds_toydataset.hdf5"

    domain_dict = {
        "type": "FrequencyDomain",
        "f_min": 0.0,
        "f_max": 100.0,
        "delta_f": 1.0,
    }
    domain = build_domain(domain_dict)
    domain_dict = domain.domain_dict
    ifos_num_asds = {"H1": 10, "L1": 8, "V1": 5}

    f = h5py.File(dataset_path, "w")
    gps_times = f.create_group("gps_times")
    grp = f.create_group("asds")
    settings = {"domain_dict": domain.domain_dict}

    for ifo, num_asds in ifos_num_asds.items():
        asds = np.sin(np.outer(np.arange(num_asds) / 100, domain()))
        grp.create_dataset(ifo, data=asds)
        gps_times[ifo] = np.arange(num_asds)

    f.attrs["settings"] = str(settings)
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
    ifos = ["H1", "V1"]
    asd_dataset = ASDDataset(dataset_path, ifos=ifos)
    # check domain
    assert asd_dataset.domain.domain_dict == domain_dict
    # check that the dataset has the correct ifos and corresponding datasets
    assert set(asd_dataset.asds.keys()) == set(ifos)
    for ifo in ifos:
        assert asd_dataset.asds[ifo].shape == (
            ifos_num_asds[ifo],
            len(asd_dataset.domain),
        )
    # check that sampling work correctly
    asd_samples = asd_dataset.sample_random_asds()
    assert set(asd_samples.keys()) == set(ifos)
    for _, asd in asd_samples.items():
        assert len(asd) == len(asd_dataset.domain)


def test_noise_dataset_update_domain(noise_dataset):
    dataset_path, domain_dict, ifos_num_asds = noise_dataset

    asd_dataset = ASDDataset(dataset_path)
    asd_samples = asd_dataset.sample_random_asds()

    delta_f = asd_dataset.domain.delta_f
    f_min = asd_dataset.domain.f_min
    f_max = asd_dataset.domain.f_max
    f_min_new = 20
    f_max_new = 40
    domain_update = {"f_min": f_min_new, "f_max": f_max_new}
    asd_dataset_truncated = ASDDataset(dataset_path, domain_update=domain_update)
    asd_samples_truncated = asd_dataset_truncated.sample_random_asds()

    domain_ref = build_domain(domain_dict)
    domain_ref.update(domain_update)
    nf_tr = len(domain_ref)
    nf = len(asd_dataset.domain)

    for ifo, n in ifos_num_asds.items():
        asd_data_full = asd_dataset.asds[ifo]
        asd_data_full_truncated = asd_dataset_truncated.asds[ifo]
        print("cp")
        # check that asd data and asd samples have the correct dimensions
        assert asd_data_full.shape == (n, nf)
        assert len(asd_samples[ifo]) == nf
        assert asd_data_full_truncated.shape == (n, nf_tr)
        assert len(asd_samples_truncated[ifo]) == nf_tr
        # check that the truncation worked properly
        assert np.all(
            asd_data_full[
                :, round(f_min_new / delta_f) : round(f_max_new / delta_f) + 1
            ]
            == asd_data_full_truncated[:, asd_dataset_truncated.domain.min_idx :]
        )
        assert not np.all(
            asd_data_full[:, round(f_min / delta_f) : round(f_min_new / delta_f)]
            == asd_data_full_truncated[
                :, round(f_min / delta_f) : asd_dataset_truncated.domain.min_idx
            ]
        )
        assert np.all(
            asd_data_full_truncated[:, : asd_dataset_truncated.domain.min_idx] == 1.0
        )
