import numpy as np
import torch


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


class DictToArray:
    def __init__(self, key):
        self.key = key

    def __call__(self, input_sample: dict):
        sample = input_sample.copy()

        d = input_sample[self.key]
        arr = np.empty(len(d), dtype=np.float32)
        for idx, v in enumerate(d.values()):
            arr[..., idx] = v

        sample[self.key] = arr
        return sample

class DictToEventArray:
    def __init__(self, key):
        self.key = key

    def __call__(self, input_sample: dict):
        sample = input_sample.copy()

        d = input_sample[self.key]
        s = (len(next(iter(d.values()))), len(d),)
        arr = np.empty(s, dtype=np.float32)
        for idx, v in enumerate(d.values()):
            arr[..., idx] = v

        sample[self.key] = arr
        return sample


class PadMask:
    """
    Pads a variable-length torch tensor to a fixed length. Also appends a binary mask
    of the same length.
    """

    def __init__(self, idx: int, dim: int, length: int):
        """
        Pads input_sample[idx][dim] to desired final length.

        Parameters
        ----------
        idx : int
            Index of the list element that needs padding.
        dim : int
            Dimension to pad.
        length : int
            Desired final length along dimension dim.
        """
        self.idx = idx
        self.dim = dim
        self.length = length

    def __call__(self, input_sample: list):
        sample = input_sample.copy()

        # Extract shape of original array
        orig_array = sample[self.idx]
        s = list(orig_array.shape)
        length_old = s[self.dim]
        sl_old = [slice(None, l) for l in s]

        # Create array padded out to desired length
        s[self.dim] = self.length
        padded_array = torch.zeros(s)
        padded_array[sl_old] += orig_array

        # Create mask
        mask = torch.ones(self.length, dtype=torch.bool)
        mask[:length_old] = torch.zeros(length_old, dtype=torch.bool)

        sample[self.idx] = padded_array
        sample.append(mask)

        return sample
