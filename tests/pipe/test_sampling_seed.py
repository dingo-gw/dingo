"""The nodes' sampling_seed property mirrors bilby_pipe's DataAnalysisInput, plus
torch: setting it seeds torch, numpy, and bilby, and None draws a seed."""

import types

import bilby
import numpy as np
import pytest
import torch

from dingo.pipe.importance_sampling import ImportanceSamplingInput
from dingo.pipe.sampling import SamplingInput


def _draws():
    return (
        torch.rand(3).numpy(),
        np.random.rand(3),
        bilby.core.utils.random.rng.random(3),
    )


@pytest.mark.parametrize("cls", [SamplingInput, ImportanceSamplingInput])
def test_sampling_seed_setter_seeds_torch_numpy_and_bilby(cls):
    obj = types.SimpleNamespace()
    cls.sampling_seed.fset(obj, 1234)
    first = _draws()
    cls.sampling_seed.fset(obj, 1234)
    assert cls.sampling_seed.fget(obj) == 1234
    assert all(np.array_equal(a, b) for a, b in zip(first, _draws()))
    cls.sampling_seed.fset(obj, None)
    assert 1 <= cls.sampling_seed.fget(obj) < 1_000_000
