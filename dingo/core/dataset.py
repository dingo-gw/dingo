import ast
import h5py
import numpy as np
import pandas as pd

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


def recursive_hdf5_load(group, keys=None):
    d = {}
    for k, v in group.items():
        if keys is None or k in keys:
            if isinstance(v, h5py.Group):
                d[k] = recursive_hdf5_load(v)
            else:
                d[k] = v[...]
                # If the array has column names, load it as a pandas DataFrame
                if d[k].dtype.names is not None:
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

    def __init__(self, file_name=None, dictionary=None, data_keys=None):
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

        # If data provided, load it
        if file_name is not None:
            self.from_file(file_name)
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

    def from_file(self, file_name):
        print("Loading dataset from " + str(file_name) + ".")
        with h5py.File(file_name, "r") as f:
            # Load only the keys that the class expects
            loaded_dict = recursive_hdf5_load(f, keys=self._data_keys)
            for k, v in loaded_dict.items():
                assert k in self._data_keys
                vars(self)[k] = v
            try:
                self.settings = ast.literal_eval(f.attrs["settings"])
            except KeyError:
                self.settings = None  # Is this necessary?

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
                vars(self)[k] = v
        if "version" not in dictionary:
            self.version = f"dingo={get_version()}"
