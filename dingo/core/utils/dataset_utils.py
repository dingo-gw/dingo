import h5py


def recursive_hdf5_save(group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            next_group = group.create_group(k)
            recursive_hdf5_save(next_group, v)
        else:
            group.create_dataset(k, data=v)


def recursive_hdf5_load(group):
    d = {}
    for k, v in group.items():
        if isinstance(v, h5py.Group):
            d[k] = {}
            recursive_hdf5_load(group)
        else:
            d[k] = v[...]
    return d


def save_dataset(dataset, settings, file_name):
    f = h5py.File(file_name, "w")
    recursive_hdf5_save(f, dataset)
    f.attrs["settings"] = str(settings)
    f.close()
