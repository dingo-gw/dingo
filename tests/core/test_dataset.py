import numpy as np
import os
from os.path import join, isfile
import types
import pytest

from dingo.core.dataset import DingoDataset
from dingo.core.utils import recursive_check_dicts_are_equal, load_data_from_file


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


def test_dataset_for_event_data(inference_setup):
    d = inference_setup
    events = list(d.data.keys())

    event = events[0]

    # first try to load data from dataset
    # this should return None as we have not saved anything to the dataset yet
    loaded_data = load_data_from_file(d.file_name, event, settings=d.settings)
    assert loaded_data is None

    if loaded_data is None:
        # for inference one would download the event data here
        data = d.data[event]
        dataset = DingoDataset(
            dictionary={event: data, "settings": d.settings}, data_keys=[event]
        )
        # save the data to file
        dataset.to_file(file_name=d.file_name, mode="a")

    # check that dataset saved correctly
    dataset = DingoDataset(file_name=d.file_name, data_keys=[event])
    assert recursive_check_dicts_are_equal(vars(dataset)[event], data)
    loaded_data = load_data_from_file(d.file_name, event, settings=d.settings)
    assert recursive_check_dicts_are_equal(loaded_data, data)
    with pytest.raises(ValueError):
        _ = load_data_from_file(d.file_name, event, settings={"bad_settings": 1})

    event = events[1]
    loaded_data = load_data_from_file(d.file_name, event)
    assert loaded_data is None

    if loaded_data is None:
        # for inference one would download the event data here
        data = d.data[event]
        dataset = DingoDataset(
            dictionary={event: data, "settings": d.settings}, data_keys=[event]
        )
        # save the data to file
        dataset.to_file(file_name=d.file_name, mode="a")

    # check that dataset saved correctly
    for idx in range(2):
        loaded_data = load_data_from_file(d.file_name, events[idx])
        assert recursive_check_dicts_are_equal(loaded_data, d.data[events[idx]])
