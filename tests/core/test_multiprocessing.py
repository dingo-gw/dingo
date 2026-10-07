"""apply_func_with_multiprocessing gives each chunk of rows its own random stream,
drawn from the caller's seed, so a Pool reproduces the serial draws exactly under
either start method (#408)."""

import multiprocessing

import bilby
import numpy as np
import pandas as pd
import pytest
import torch

import dingo.core.multiprocessing
from dingo.core.multiprocessing import CHUNK_SIZE, apply_func_with_multiprocessing


def _draws(row=None):
    """One draw from each global generator a task may use."""
    return (
        bilby.core.utils.random.rng.uniform(),
        np.random.uniform(),
        torch.rand(1).item(),
    )


def _seed_caller():
    np.random.seed(0)
    bilby.core.utils.random.seed(0)
    torch.manual_seed(0)


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_pool_reproduces_serial_draws(monkeypatch, start_method):
    theta = pd.DataFrame({"x": np.arange(3 * CHUNK_SIZE + 5, dtype=float)})
    _seed_caller()
    serial = apply_func_with_multiprocessing(_draws, theta, num_processes=1)
    serial_after = _draws()

    monkeypatch.setattr(
        dingo.core.multiprocessing,
        "Pool",
        multiprocessing.get_context(start_method).Pool,
    )
    _seed_caller()
    pooled = apply_func_with_multiprocessing(_draws, theta, num_processes=3)

    assert np.array_equal(serial, pooled)
    # No two rows share a draw from any generator.
    assert all(len(np.unique(draws)) == len(theta) for draws in serial.T)
    # The serial path leaves the caller's generators as a Pool does.
    assert _draws() == serial_after


def test_consecutive_calls_draw_different_streams():
    theta = pd.DataFrame({"x": np.zeros(4)})
    first = apply_func_with_multiprocessing(_draws, theta)
    second = apply_func_with_multiprocessing(_draws, theta)
    assert not np.any(first == second)


def test_tuple_of_arrays_rows():
    values = np.arange(2 * CHUNK_SIZE + 1)
    result = apply_func_with_multiprocessing(np.multiply, (values, -values))
    assert np.array_equal(result, -(values**2))
