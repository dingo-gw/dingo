from typing import List, Optional, Union
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
        elif isinstance(v, (int, float, complex, str, list)):
            # TODO: Set scalars as attributes?
            group.create_dataset(k, data=v)
        else:
            raise TypeError(f"Cannot save datatype {type(v)} as hdf5 dataset.")


def recursive_hdf5_load(
    group,
    keys: Optional[List[str]] = None,
    idx: Optional[Union[int, List[int]]] = None,
    dtype_map: Optional[dict] = None,
    _inherited_dtype=None,
):
    """This is a generic helper function to recursively load data from an HDF5 file.

    Parameters
    ----------
    group: h5py.Group
        Group from which to recursively load data.
    keys: list[str] or None
        List of keys to load. If None, load all keys.
    idx: int or list[int] or None
        If idx is provided, only the datapoints corresponding to the given indices are loaded.
    dtype_map: dict or None
        Mapping from group/dataset names to target numpy dtypes or nested dicts.
        This enables direct dtype conversion during HDF5 read, avoiding intermediate
        memory allocation when changing precision. Values can be:
        - A numpy dtype: applies to all arrays within that group (recursively)
        - A nested dict: specifies dtypes for children of that group
        E.g., {"polarizations": np.complex64} converts all arrays under "polarizations"
        to complex64.
        E.g., {"svd": {"V": np.complex64, "s": np.float32}} converts V and s separately.
    _inherited_dtype:
        Internal parameter for propagating dtype through recursive calls.
    """
    d = {}
    for k, v in group.items():
        if keys is None or k in keys:
            # Check if this key has a dtype or nested map specified
            dtype_spec = dtype_map.get(k) if dtype_map else None

            if isinstance(v, h5py.Group):
                if isinstance(dtype_spec, dict):
                    # Nested dict: use as dtype_map for this subgroup
                    d[k] = recursive_hdf5_load(v, idx=idx, dtype_map=dtype_spec)
                else:
                    # dtype or None: use as inherited_dtype for children
                    effective_dtype = dtype_spec if dtype_spec is not None else _inherited_dtype
                    d[k] = recursive_hdf5_load(
                        v, idx=idx, dtype_map=dtype_map, _inherited_dtype=effective_dtype
                    )
            else:
                # For datasets, dtype_spec should be a dtype (not a dict)
                effective_dtype = dtype_spec if dtype_spec is not None else _inherited_dtype
                if isinstance(effective_dtype, dict):
                    raise TypeError(
                        f"dtype_map specifies a dict for dataset '{k}', but dicts are "
                        f"only valid for groups. Use a numpy dtype instead."
                    )

                # Use astype view for direct dtype conversion during load if specified.
                # Skip for structured arrays (they're converted to DataFrames later).
                if (
                    effective_dtype is not None
                    and v.dtype != effective_dtype
                    and v.dtype.names is None
                ):
                    view = v.astype(effective_dtype)
                else:
                    view = v

                # Load values from hdf5 file as np.ndarray
                if idx is None:
                    # Load all values
                    d[k] = view[...]
                elif isinstance(idx, list) and len(idx) > 1:
                    # Load batch of indices: hdf5 load requires index list to be sorted
                    sorting = np.argsort(idx)
                    sorted_idx = np.array(idx)[sorting]
                    reverse_sorting = np.zeros_like(sorting)
                    reverse_sorting[sorting] = np.arange(len(sorting))
                    d[k] = view[sorted_idx][reverse_sorting]
                else:
                    # Load specific idx
                    d[k] = view[idx]
                # Update data types
                # If the array has column names, load it as a pandas DataFrame
                if d[k].dtype.names is not None:
                    d[k] = pd.DataFrame(d[k])
                    # Apply dtype conversion to DataFrame if specified
                    if effective_dtype is not None:
                        d[k] = d[k].astype(effective_dtype, copy=False)
                # Convert 0-dimensional arrays to scalars
                elif d[k].ndim == 0:
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
        self,
        file_name: Optional[str] = None,
        dictionary: Optional[dict] = None,
        data_keys: Optional[List] = None,
        leave_on_disk_keys: Optional[list] = None,
        dtype_map: Optional[dict] = None,
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
        leave_on_disk_keys: Optional[list]
            Keys for which the values are not loaded into RAM when initializing the dataset.
            This reduces the memory footprint during training. Instead, the values are
            loaded from the HDF5 file during training.
        dtype_map: Optional[dict]
            Mapping from group names to target numpy dtypes for loading. Passed to
            recursive_hdf5_load to enable direct dtype conversion during HDF5 read,
            avoiding intermediate memory allocation.
        """
        self._data_keys = list(data_keys)  # Make a copy before modifying.
        self._data_keys.append("version")

        # Ensure all potential variables have None values to begin
        for key in self._data_keys:
            vars(self)[key] = None
        self.settings = None
        if leave_on_disk_keys is None:
            leave_on_disk_keys = []
        self._leave_on_disk_keys = leave_on_disk_keys

        # If data provided, load it
        if file_name is not None:
            self.from_file(file_name, dtype_map=dtype_map)
        elif dictionary is not None:
            self.from_dictionary(dictionary)

    def to_file(self, file_name: str, mode: str = "w"):
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

    def from_file(self, file_name: str, dtype_map: Optional[dict] = None):
        print(f"Loading dataset from {str(file_name)}.")
        if self._leave_on_disk_keys:
            print(f"Omitting data keys {self._leave_on_disk_keys}.")

        with h5py.File(file_name, "r") as f:
            loaded_dict = recursive_hdf5_load(
                f,
                keys=[k for k in self._data_keys if k not in self._leave_on_disk_keys],
                dtype_map=dtype_map,
            )
            # Set the keys that the class expects
            for k, v in loaded_dict.items():
                assert k in self._data_keys
                setattr(self, k, v)
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
                setattr(self, k, v)
        if "version" not in dictionary:
            self.version = f"dingo={get_version()}"
