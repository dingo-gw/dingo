import ast
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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
            raise TypeError('Cannot save datatype {} as hdf5 dataset.'.format(type(v)))


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


class DingoDataset(Dataset):

    def __init__(self):
        self._save_keys = []
        self.settings = None

    def save(self, file_name):
        print('Saving dataset to ' + file_name)
        save_dict = {k: v for k, v in vars(self).items() if k in self._save_keys}
        f = h5py.File(file_name, "w")
        recursive_hdf5_save(f, save_dict)
        f.attrs["settings"] = str(self.settings)
        f.close()

    def load(self, file_name):
        print('Loading dataset from ' + file_name + ' :')
        f = h5py.File(file_name, "r")
        loaded_dict = recursive_hdf5_load(f)
        for k, v in loaded_dict.items():
            if k in self._save_keys:  # Load only the keys that the class expects
                print('  ' + k)
                vars(self)[k] = v
        try:
            self.settings = ast.literal_eval(f.attrs["settings"])
        except KeyError:
            settings = None
        f.close()


class TestDataset(DingoDataset):

    def __init__(self):
        super().__init__()
        self._save_keys = ['parameters', 'polarizations']

    def __getitem__(self, item):
        pass


