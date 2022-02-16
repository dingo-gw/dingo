import ast
from os.path import isfile
import h5py
import numpy as np
import pandas as pd


def recursive_hdf5_save(group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            next_group = group.create_group(k)
            recursive_hdf5_save(next_group, v)
        elif isinstance(v, np.ndarray):
            group.create_dataset(k, data=v)
        elif isinstance(v, pd.DataFrame):
            group.create_dataset(k, data=v.to_records(index=False))
        else:
            raise TypeError("Cannot save datatype {} as hdf5 dataset.".format(type(v)))


def recursive_hdf5_load(group):
    d = {}
    for k, v in group.items():
        if isinstance(v, h5py.Group):
            d[k] = recursive_hdf5_load(v)
        else:
            d[k] = v[...]
            # If the array has column names, load it as a pandas DataFrame
            if d[k].dtype.names is not None:
                d[k] = pd.DataFrame(d[k])
    return d


class DingoDataset:
    """This is a generic dataset class with save / load methods.

    A common use case is to inherit multiply from DingoDataset and
    torch.utils.data.Dataset, in which case the subclass picks up these I/O methods,
    and DingoDataset is acting as a Mixin class.

    Alternatively, if the torch Dataset is not needed, then DingoDataset can be
    subclassed directly.
    """

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
        # Ensure all potential variables have None values to begin
        for key in data_keys:
            vars(self)[key] = None
        self._data_keys = data_keys
        self.settings = None

        # If data provided, load it
        if file_name is not None:
            self.from_file(file_name)
        elif dictionary is not None:
            self.from_dictionary(dictionary)

    def to_file(self, file_name):
        print("Saving dataset to " + file_name)
        save_dict = {
            k: v
            for k, v in vars(self).items()
            if k in self._data_keys and v is not None
        }
        f = h5py.File(file_name, "w")
        recursive_hdf5_save(f, save_dict)
        f.attrs["settings"] = str(self.settings)
        f.close()

    def from_file(self, file_name):
        print("\nLoading dataset from " + file_name + ".")
        f = h5py.File(file_name, "r")
        loaded_dict = recursive_hdf5_load(f)
        for k, v in loaded_dict.items():
            if k in self._data_keys:  # Load only the keys that the class expects
                # print("  " + k)
                vars(self)[k] = v
        try:
            self.settings = ast.literal_eval(f.attrs["settings"])
        except KeyError:
            self.settings = None  # Is this necessary?
        f.close()

    def append_to_file(self, file_name):
        """
        Appends the dataset to a file.
        * If isfile(file_name):
            -> check that settings are identical (if not: raise ValueError)
            -> check that keys between dataset and saved file don't overlap
               (if not: raise Exception)
            -> append dataset
        * If not isfile(file_name):
            -> write dataset to new file

        Parameters
        ----------
        file_name: str
            name of the dataset file
        """
        if not isfile(file_name):
            self.to_file(file_name)

        else:
            # perform checks
            with h5py.File(file_name, "r") as f:
                # check that self.settings is identical to saved settings
                try:
                    saved_settings = ast.literal_eval(f.attrs["settings"])
                except KeyError:
                    saved_settings = None
                if self.settings != saved_settings:
                    raise ValueError(
                        f"Settings incompatible with saved settings, "
                        f"{self.settings} != {saved_settings}."
                    )

                # check that keys don't overlap
                if len(set(self._data_keys) & set(f.keys())) > 0:
                    raise Exception(
                        f"Overlapping keys between dataset and saved file: "
                        f"{set(self._data_keys) & set(f.files)}."
                    )

            # append dataset to save file
            with h5py.File(file_name, "a") as f:
                print(f"Appending dataset to {file_name}")
                save_dict = {
                    k: v
                    for k, v in vars(self).items()
                    if k in self._data_keys and v is not None
                }
                recursive_hdf5_save(f, save_dict)

    def to_dictionary(self):
        dictionary = {
            k: v
            for k, v in vars(self).items()
            if (k in self._data_keys or k == "settings") and v is not None
        }
        return dictionary

    def from_dictionary(self, dictionary):
        for k, v in dictionary.items():
            if k in self._data_keys or k == "settings":
                vars(self)[k] = v


def load_data_from_file(file_name, data_key):
    """
    Loads data from a file.

    * If isfile(file_name):
        -> load file as DingoDataset
        -> return vars(dataset)[data_key] (which is None, if the key does not match)
    * else:
        -> return None

    Parameters
    ----------
    file_name: str
        name of the dataset file

    data_key:
        key for data in dataset file

    Returns
    -------

    """
    if not isfile(file_name):
        return None

    else:
        dataset = DingoDataset(file_name=file_name, data_keys=[data_key])
        return vars(dataset)[data_key]
