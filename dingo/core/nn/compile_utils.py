"""``torch.compile`` support for the flow network (``local.torch_compile``).

The neural spline flow launches tens of thousands of tiny CUDA kernels per step
(one per spline/indexing op in every coupling transform), so on a modern GPU the
step is bound by kernel-launch overhead rather than by arithmetic. ``torch.compile``
fuses these kernels and removes most of that overhead.

This only works if the rational-quadratic spline in ``glasflow.nflows`` is written
with static shapes. The original implementation selects the points inside/outside
the spline tails with data-dependent boolean-mask indexing and a host-syncing
``torch.any`` branch, which force graph breaks and stop the kernels from fusing;
with it, ``torch.compile`` is a net *slowdown*. A numerically identical static-shape
rewrite lives in the ``compile-friendly-rqs`` branch of
https://github.com/nihargupte-ph/nflows (vendored by the matching branch of
https://github.com/nihargupte-ph/glasflow) until it is merged upstream.
"""

import contextlib
import inspect
import os
import tempfile

import torch

GLASFLOW_INSTALL_HINT = (
    "pip install git+https://github.com/nihargupte-ph/glasflow@compile-friendly-rqs"
)


def spline_is_compile_friendly() -> bool:
    """Whether the installed glasflow ships the static-shape RQ spline.

    The rewrite added a ``check_domain`` argument to ``rational_quadratic_spline``
    (so the clamped call can skip its host-syncing domain check); its presence is
    used as the marker.
    """
    from glasflow.nflows.transforms.splines import rational_quadratic as rq

    return "check_domain" in inspect.signature(rq.rational_quadratic_spline).parameters


@contextlib.contextmanager
def eager_mode():
    """Run compiled networks eagerly inside the block, without (re)compiling.

    The test epoch runs the network in eval mode, which would otherwise trigger a
    second full compilation (several minutes for a production-size network) that
    a test epoch of a few hundred steps never amortizes.
    """
    set_stance = getattr(torch.compiler, "set_stance", None)
    if set_stance is None:  # torch < 2.6
        yield
        return
    with set_stance("force_eager"):
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

    Raises
    ------
    RuntimeError
        If the installed glasflow lacks the static-shape spline, since compiling
        would then be slower than eager mode.
    """
    if not spline_is_compile_friendly():
        raise RuntimeError(
            "torch_compile requires a glasflow with the static-shape "
            "rational-quadratic spline; the installed version would compile with "
            "graph breaks and run slower than eager mode. Install it with\n"
            f"    {GLASFLOW_INSTALL_HINT}\n"
            "or set torch_compile: false."
        )
    if rank is not None:
        base = cache_dir or tempfile.gettempdir()
        rank_cache = os.path.join(base, f"dingo_inductor_rank{rank}")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = rank_cache
        os.environ["TRITON_CACHE_DIR"] = os.path.join(rank_cache, "triton")
    return torch.compile(network)
