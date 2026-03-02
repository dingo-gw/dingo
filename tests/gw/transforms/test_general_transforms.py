import numpy as np
import pytest

from dingo.gw.transforms import UnpackDict


def test_UnpackDict():
    sample = {"a": 10, "b": np.random.rand(100), "c": None}
    unpack_dict = UnpackDict(["b", "a"])
    b, a = unpack_dict(sample)
    assert id(b) == id(sample["b"])
    assert a == sample["a"]
