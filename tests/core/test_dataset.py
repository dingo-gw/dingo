"""Tests for the recursive HDF5 save / load helpers in dingo.core.dataset."""

import h5py
import numpy as np
import pytest

from dingo.core.dataset import recursive_hdf5_load, recursive_hdf5_save


@pytest.mark.parametrize(
    "value",
    [
        [59.5, 60.625],
        [[59.5, 60.5], [119.0, 121.0]],
        [1, 2, 3],
        [True, False],
        ["H1", "L1"],
        ["H1"],
        [["H1", "L1"], ["V1", "K1"]],
        [],
    ],
)
def test_lists_survive_a_round_trip(tmp_path, value):
    with h5py.File(tmp_path / "f.hdf5", "w") as f:
        recursive_hdf5_save(f, {"group": {"value": value}})
    with h5py.File(tmp_path / "f.hdf5", "r") as f:
        loaded = recursive_hdf5_load(f)["group"]["value"]
    assert loaded == value
    assert type(loaded) is list


def test_lists_survive_a_round_trip_with_dtype_map(tmp_path):
    with h5py.File(tmp_path / "f.hdf5", "w") as f:
        recursive_hdf5_save(f, {"value": [1.0, 2.0]})
    with h5py.File(tmp_path / "f.hdf5", "r") as f:
        loaded = recursive_hdf5_load(f, dtype_map={"value": np.float32})["value"]
    assert loaded == [1.0, 2.0]


def test_untagged_datasets_load_as_before(tmp_path):
    # Files written before lists were tagged: numeric lists come back as arrays,
    # 1-D string lists as lists of str.
    with h5py.File(tmp_path / "f.hdf5", "w") as f:
        f.create_dataset("numbers", data=[[59.5, 60.5]])
        f.create_dataset("strings", data=["H1", "L1"])
    with h5py.File(tmp_path / "f.hdf5", "r") as f:
        loaded = recursive_hdf5_load(f)
    assert isinstance(loaded["numbers"], np.ndarray)
    assert loaded["strings"] == ["H1", "L1"]
