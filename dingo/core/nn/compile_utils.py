"""
The neural spline flow launches tens of thousands of tiny CUDA kernels per step
(one per spline/indexing op in every coupling transform), so on a modern GPU the
step is bound by kernel-launch overhead rather than by arithmetic. ``torch.compile``
fuses these kernels and removes most of that overhead.
"""

import contextlib
import os
import tempfile

import torch


@contextlib.contextmanager
def eager_mode():
    """Run compiled networks eagerly inside the block, without (re)compiling.

    The test epoch runs the network in eval mode. This is a different graph than 
    what you would see at training. Therefore, it would trigger a
    second full compilation. Instead, we do eager evaluation (not using the 
    fused kernels) to avoid the test loop taking a long time. We could also 
    compile a "test time" graph, but it does not amortize as well as training time. 
    """
    with torch.compiler.set_stance("force_eager"):
        yield


def compile_network(
    network: torch.nn.Module, rank: int = None, cache_dir: str = None
) -> torch.nn.Module:
    """Return ``torch.compile(network)``, set up for (optionally) DDP training.

    Parameters
    ----------
    network : torch.nn.Module
        The network to compile. Under DDP pass the DDP-wrapped network, so that
        the gradient all-reduce can still overlap with the backward pass.
    rank : int, optional
        DDP rank. When given, each rank is pointed at its own on-disk
        Inductor/Triton cache: the ranks compile concurrently and would otherwise
        race on the shared cache files.
    cache_dir : str, optional
        Base directory for the per-rank caches (default: the system temp dir). It
        must be **node-local**: Triton shared objects written to a network
        filesystem can be unloadable from another process, which hangs the run.
    """
    if rank is not None:
        # in case the default cache dir is not available
        base = cache_dir or tempfile.gettempdir()
        # naming the cache so that different DDP ranks do not interferej
        rank_cache = os.path.join(base, f"dingo_inductor_rank{rank}")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = rank_cache
        os.environ["TRITON_CACHE_DIR"] = os.path.join(rank_cache, "triton")
    return torch.compile(network)
