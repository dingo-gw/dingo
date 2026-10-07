"""Apply a function to each row of a table, in parallel and reproducibly."""

from multiprocessing import Pool

import bilby
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from dingo.core.utils.torchutils import seed_generators

# Rows per Pool task. Each chunk of rows draws from its own random stream, so the
# draws depend on the seed and on CHUNK_SIZE, but not on the number of processes.
# Seeding takes ~0.5 ms, once per chunk (dominated by bilby.core.utils.random.seed()).
CHUNK_SIZE = 32

# The function applied by a Pool worker, set once per worker by the initializer so
# that it is not pickled with every chunk.
_func = None


def apply_func_with_multiprocessing(func: callable, theta, num_processes: int = 1):
    """
    Apply func to each row of theta, optionally with multiprocessing.

    The rows are split into chunks of CHUNK_SIZE. Before evaluating a chunk, numpy,
    bilby and torch (CPU) are seeded from that chunk's child of a SeedSequence, whose
    entropy is drawn from the caller's numpy generator. Random draws inside func are
    therefore independent across chunks, determined by the caller's seed, and the
    same for any num_processes. The serial path restores the caller's generators
    afterwards.

    Parameters
    ----------
    func: callable
        Called as func(row_dict) for a DataFrame, or func(*row) for a tuple of arrays.
    theta : pd.DataFrame or tuple of np.ndarray
        Rows to evaluate: the rows of a DataFrame, or the rows of equally long arrays
        taken together.
    num_processes : int
        Number of parallel processes to use.

    Returns
    -------
    result: np.ndarray
        Output array, where result[idx] = func applied to row idx.
    """
    if isinstance(theta, pd.DataFrame):
        num_rows = len(theta)
        chunks = [
            theta.iloc[i : i + CHUNK_SIZE] for i in range(0, num_rows, CHUNK_SIZE)
        ]
    else:
        num_rows = len(theta[0])
        chunks = [
            tuple(a[i : i + CHUNK_SIZE] for a in theta)
            for i in range(0, num_rows, CHUNK_SIZE)
        ]
    # Drawn from the caller's generator, so that consecutive calls use different
    # streams and a seeded caller gets the same streams every time.
    entropy = np.random.randint(2**32, size=4).tolist()
    tasks = [(entropy, index, chunk) for index, chunk in enumerate(chunks)]

    num_processes = min(num_processes, len(tasks))
    if num_processes > 1:
        with Pool(
            processes=num_processes, initializer=_init_worker, initargs=(func,)
        ) as pool:
            results = pool.starmap(_apply_to_chunk, tasks, chunksize=1)
    else:
        numpy_state = np.random.get_state()
        bilby_rng = bilby.core.utils.random.rng
        torch_state = torch.get_rng_state()
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                results = [_apply_to_chunk(*task, func=func) for task in tasks]
        finally:
            np.random.set_state(numpy_state)
            bilby.core.utils.random.seed(bilby_rng)
            torch.set_rng_state(torch_state)

    return np.array([r for chunk_result in results for r in chunk_result])


def _init_worker(func):
    """Pool initializer: store the function to apply and limit BLAS to one thread."""
    global _func
    _func = func
    threadpool_limits(limits=1, user_api="blas")


def _apply_to_chunk(entropy, index, rows, func=None):
    """Seed numpy, bilby and torch (CPU) from the stream of chunk `index`, then apply
    func (by default the worker's) to each row of the chunk."""
    seed_generators(np.random.SeedSequence(entropy, spawn_key=(index,)))
    func = _func if func is None else func
    if isinstance(rows, pd.DataFrame):
        return [func(row) for row in rows.to_dict("records")]
    return [func(*row) for row in zip(*rows)]
