"""The nodes' sampling_seed property mirrors bilby_pipe's DataAnalysisInput, plus
torch: setting it seeds torch, numpy, and bilby, and None draws a seed. The generators
get separate streams, also between the sampling and importance-sampling jobs."""

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
    assert isinstance(cls.sampling_seed.fget(obj), int)


def test_sampling_and_importance_sampling_jobs_draw_different_streams():
    obj = types.SimpleNamespace()
    SamplingInput.sampling_seed.fset(obj, 1234)
    sampling = _draws()
    ImportanceSamplingInput.sampling_seed.fset(obj, 1234)
    importance_sampling = _draws()
    assert not any(np.array_equal(a, b) for a, b in zip(sampling, importance_sampling))


def test_torch_and_numpy_streams_do_not_overlap():
    """torch's CPU generator and numpy's legacy one are both MT19937: seeded with the
    same integer, they emit the same 32-bit words."""
    SamplingInput.sampling_seed.fset(types.SimpleNamespace(), 1234)
    torch_words = torch.randint(0, 2**31, (100,)).numpy()
    numpy_words = np.random.randint(0, 2**31, 100)
    assert np.intersect1d(torch_words, numpy_words).size == 0
