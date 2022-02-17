import numpy as np
import os
from os.path import join, isfile
import types
import pytest

from dingo.core.dataset import DingoDataset, load_data_from_file


@pytest.fixture()
def inference_setup():
    d = types.SimpleNamespace()

    # specify file name for dataset and remove it if it exists
    tmp_dir = "./tmp_files"
    os.makedirs(tmp_dir, exist_ok=True)
    d.file_name = join(tmp_dir, "event_data.hdf5")
    if isfile(d.file_name):
        os.remove(d.file_name)

    d.data = {
        str(10): {"strain": {"H": np.ones(5), "L": np.zeros(5)}, "asd": np.ones(2)},
        str(20): {"strain": {"H": np.ones(5) * 2, "L": np.zeros(5)}, "asd": np.ones(2)},
        str(30): {"strain": {"H": np.ones(5) * 3, "L": np.zeros(5)}, "asd": np.ones(2)},
    }

    d.settings = {"data": "dummy_settings"}

    return d


def recursive_check_dicts_are_equal(dict_a, dict_b):
    if dict_a.keys() != dict_b.keys():
        return False
    else:
        for k, v_a in dict_a.items():
            v_b = dict_b[k]
            if type(v_a) != type(v_b):
                return False
            if type(v_a) == dict:
                if not recursive_check_dicts_are_equal(v_a, v_b):
                    return False
            elif not np.all(v_a == v_b):
                return False
    return True


def test_dataset_for_event_data(inference_setup):
    d = inference_setup
    events = list(d.data.keys())

    event = events[0]

    # first try to load data from dataset
    # this should return None as we have not saved anything to the dataset yet
    loaded_data = load_data_from_file(d.file_name, event)
    assert loaded_data is None

    if loaded_data is None:
        # for real inference one would download the event data here
        data = d.data[event]
        # save the data to file
        dataset = DingoDataset(
            dictionary={event: data, "settings": d.settings}, data_keys=[event]
        )
        dataset.to_file(file_name=d.file_name, mode="a")

    # check that dataset saved correctly
    dataset = DingoDataset(file_name=d.file_name, data_keys=[event])
    assert recursive_check_dicts_are_equal(vars(dataset)[event], data)
    loaded_data = load_data_from_file(d.file_name, event)
    assert recursive_check_dicts_are_equal(loaded_data, data)

    # # check that error is raised if one tries to append the dataset again to the file
    # with pytest.raises(Exception):
    #     dataset.to_file(file_name=d.file_name, mode="a")

    event = events[1]
    loaded_data = load_data_from_file(d.file_name, event)
    assert loaded_data is None

    if loaded_data is None:
        # for real inference one would download the event data here
        data = d.data[event]
        dataset = DingoDataset(dictionary={event: data}, data_keys=[event])
        # check that ValueError is raised if settings are off (they are None here)
        # with pytest.raises(ValueError):
        #     dataset.append_to_file(file_name=d.file_name)
        # save the data to file
        dataset.settings = d.settings
        dataset.to_file(file_name=d.file_name, mode="a")

    # check that dataset saved correctly
    for idx in range(2):
        loaded_data = load_data_from_file(d.file_name, events[0])
        assert recursive_check_dicts_are_equal(loaded_data, d.data[events[0]])
