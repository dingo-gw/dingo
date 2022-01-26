import ast
import h5py

from dingo.core.dataset import recursive_hdf5_save, recursive_hdf5_load

# This should eventually be removed. Most usages have been taken into the DingoDataset
# class.


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