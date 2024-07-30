from typing import Tuple
import ast
import h5py
import numpy as np
import pandas as pd

import ctypes
import multiprocessing.sharedctypes

from dingo.core.utils.misc import get_version


def recursive_hdf5_save(group, d):
    for k, v in d.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            next_group = group.create_group(k)
            recursive_hdf5_save(next_group, v)
        elif isinstance(v, np.ndarray):
            group.create_dataset(k, data=v)
        elif isinstance(v, pd.DataFrame):
            group.create_dataset(k, data=v.to_records(index=False))
        elif isinstance(v, (int, float, str, list)):
            # TODO: Set scalars as attributes?
            group.create_dataset(k, data=v)
        else:
            raise TypeError(f"Cannot save datatype {type(v)} as hdf5 dataset.")


def recursive_hdf5_load(group, keys=None, leave_on_disk_keys=None, idx: Tuple[int,...] = None):
    d = {}
    for k, v in group.items():
        if keys is None or k in keys:
            if isinstance(v, h5py.Group):
                d[k] = recursive_hdf5_load(v, leave_on_disk_keys=leave_on_disk_keys, idx=idx)
            else:
                if leave_on_disk_keys is not None and k in leave_on_disk_keys:
                    # Insert dummy value into dict
                    d[k] = None
                else:
                    # Load complete array or only specific idx
                    if idx is None or v.shape == ():
                        d[k] = v[...]
                    else:
                        d[k] = v[idx]
                    # If the array has column names, convert it to a pandas DataFrame
                    if d[k].dtype.names is not None:
                        # Convert row v[idx] into list for pd
                        if type(d[k]) == np.void:
                            d[k] = pd.DataFrame([list(d[k])], columns=d[k].dtype.names)
                        else:
                            d[k] = pd.DataFrame(d[k])
                    # Convert arrays of size 1 to scalars
                    elif d[k].size == 1:
                        d[k] = d[k].item()
                        if isinstance(d[k], bytes):
                            # Assume this is a string.
                            d[k] = d[k].decode()
                    # If an array is 1D and of type object, assume it originated as a list
                    # of strings.
                    elif d[k].ndim == 1 and d[k].dtype == "O":
                        d[k] = [x.decode() for x in d[k]]
    return d


class DingoDataset:
    """This is a generic dataset class with save / load methods.

    A common use case is to inherit multiply from DingoDataset and
    torch.utils.data.Dataset, in which case the subclass picks up these I/O methods,
    and DingoDataset is acting as a Mixin class.

    Alternatively, if the torch Dataset is not needed, then DingoDataset can be
    subclassed directly.
    """

    dataset_type = "dingo_dataset"

    def __init__(
        self, file_name=None, dictionary=None, data_keys=None, leave_on_disk_keys=None
    ):
        """
        For constructing, provide either file_name, or dictionary containing data and
        settings entries, or neither.

        Parameters
        ----------
        file_name : str
            HDF5 file containing a dataset
        dictionary : dict
            Contains settings and data entries. The data keys should be the same as
            save_keys
        data_keys : list
            Variables that should be saved / loaded. This allows for class to store
            additional variables beyond those that are saved. Typically, this list
            would be provided by any subclass.
        """
        self._data_keys = list(data_keys)  # Make a copy before modifying.
        self._data_keys.append("version")

        # Ensure all potential variables have None values to begin
        for key in self._data_keys:
            vars(self)[key] = None
        self.settings = None
        self.version = None
        self.leave_on_disk_keys = leave_on_disk_keys

        # If data provided, load it
        if file_name is not None:
            self.from_file(file_name, leave_on_disk_keys)
        elif dictionary is not None:
            self.from_dictionary(dictionary)

    def to_file(self, file_name, mode="w"):
        print("Saving dataset to " + str(file_name))
        save_dict = {
            k: v
            for k, v in vars(self).items()
            if k in self._data_keys and v is not None
        }
        with h5py.File(file_name, mode) as f:
            recursive_hdf5_save(f, save_dict)
            if self.settings:
                f.attrs["settings"] = str(self.settings)
            if self.dataset_type:
                f.attrs["dataset_type"] = self.dataset_type

                # Explicit line for loading and closing, only close it if there's nothing left on disk

    def from_file(self, file_name, leave_on_disk_keys=None):
        print("Loading dataset from " + str(file_name) + ".")
        if leave_on_disk_keys is None or leave_on_disk_keys == []:
            with h5py.File(file_name, "r") as f:
                # Load only the keys that the class expects
                loaded_dict = recursive_hdf5_load(f, keys=self._data_keys)
                for k, v in loaded_dict.items():
                    assert k in self._data_keys
                    setattr(self, k, v)
                try:
                    self.settings = ast.literal_eval(f.attrs["settings"])
                except KeyError:
                    self.settings = None  # Is this necessary?
        else:
            # Open hdf5 file
            f = h5py.File(file_name, "r")
            # Load only the keys that the class expects
            loaded_dict = recursive_hdf5_load(
                f, keys=self._data_keys, leave_on_disk_keys=leave_on_disk_keys
            )
            for k, v in loaded_dict.items():
                assert k in self._data_keys
                setattr(self, k, v)
            try:
                self.settings = ast.literal_eval(f.attrs["settings"])
            except KeyError:
                self.settings = None  # Is this necessary?
            # The hdf5 file is not closed here since the values corresponding to the leave_on_disk_keys are only loaded
            # by reference at this point. They will be loaded on the fly in WaveformDataset.__getitem__()
            # The file should be closed when the DataLoader enters the garbage collection. Does this happen
            # automatically?

    def to_dictionary(self):
        dictionary = {
            k: v
            for k, v in vars(self).items()
            if (k in self._data_keys or k == "settings") and v is not None
        }
        return dictionary

    def from_dictionary(self, dictionary: dict):
        for k, v in dictionary.items():
            if k in self._data_keys or k == "settings":
                setattr(self, k, v)
        if "version" not in dictionary:
            self.version = f"dingo={get_version()}"
