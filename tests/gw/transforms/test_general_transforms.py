import pytest
import numpy as np

from dingo.gw.transforms import SelectKeys, UnpackDict

def test_UnpackDict():
    sample = {'a': 10, 'b': np.random.rand(100), 'c': None}
    unpack_dict = UnpackDict(['b', 'a'])
    b, a = unpack_dict(sample)
    assert id(b) == id(sample['b'])
    assert a == sample['a']


def test_SelectKeys():
    sample = {'a': 10, 'b': np.random.rand(100), 'c': None}
    select_keys = SelectKeys(['b', 'a'])
    out = select_keys(sample)
    assert list(out) == ['b', 'a']
    assert id(out['b']) == id(sample['b'])
    assert out['a'] == sample['a']
    with pytest.raises(KeyError, match="missing keys"):
        SelectKeys(['a', 'd'])(sample)