import ast

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


def save_dataset(dataset, settings, file_name):
    print('Saving dataset to ' + file_name)
    f = h5py.File(file_name, "w")
    recursive_hdf5_save(f, dataset)
    f.attrs["settings"] = str(settings)
    f.close()


def load_dataset(file_name):
    f = h5py.File(file_name, "r")
    data = recursive_hdf5_load(f)
    try:
        settings = ast.literal_eval(f.attrs["settings"])
    except KeyError:
        settings = None
    f.close()
    return data, settings