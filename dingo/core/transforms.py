from copy import deepcopy

import numpy as np


class AdjustNumpyPrecision:
    def __init__(self, dtypes_dict):
        self.dtypes_dict = dtypes_dict

    def __call__(self, input_sample: dict):
        sample = input_sample.copy()
        for input_key, dtype in self.dtypes_dict.items():
            data = sample[input_key]
            if isinstance(data, np.ndarray):
                sample[input_key] = data.astype(dtype)
            elif isinstance(data, dict):
                # allow for one nested level
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = v.astype(dtype)
                    else:
                        raise NotImplementedError()
            else:
                raise NotImplementedError()
        return sample


class GetItem:
    def __init__(self, key):
        self.key = key

    def __call__(self, sample):
        return sample[self.key]


class RenameKey:
    def __init__(self, old, new):
        self.old = old
        self.new = new

    def __call__(self, input_sample: dict):
        sample = input_sample.copy()
        sample[self.new] = sample.pop(self.old)
        return sample


class Copy:
    def __init__(self, source_key, destination_key):
        # for nested dicts, separate levels with "/", key_level0/key_level1/...
        self.source_key = source_key
        self.destination_key = destination_key

    def __call__(self, input_sample: dict):
        sample = deepcopy(input_sample)  # deepcopy needed for nested dicts
        source = sample
        for key_level in self.source_key.split("/"):
            source = source[key_level]
        destination = sample
        for key_level in self.destination_key.split("/")[:-1]:
            destination = destination[key_level]
        if self.destination_key.split("/")[-1] in destination:
            raise ValueError(f"Destination key {self.destination_key} already present.")
        destination[self.destination_key.split("/")[-1]] = source
        return sample
