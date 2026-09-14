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


def is_compiled(network: torch.nn.Module) -> bool:
    """True if ``network`` is (or wraps) a ``torch.compile``-d module."""
    while network is not None:
        if hasattr(network, "_orig_mod"):  # torch.compile OptimizedModule
            return True
        network = getattr(network, "module", None)  # DDP
    return False


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
        Base directory for the on-disk Inductor/Triton cache (default: the system
        temp dir). It must be **node-local**: Triton shared objects written to a
        network filesystem can be unloadable from another process, which hangs
        the run.
    """
    if rank is not None or cache_dir is not None:
        base = cache_dir or tempfile.gettempdir()
        name = "dingo_inductor" if rank is None else f"dingo_inductor_rank{rank}"
        cache = os.path.join(base, name)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    return torch.compile(network)


def reset_graphs_if_requires_grad_changes(
    network: torch.nn.Module, name_contains: str, requires_grad: bool
) -> bool:
    """Discard the compiled graphs of ``network`` if setting ``requires_grad`` on the
    parameters whose name contains ``name_contains`` would change their state.

    Dynamo does not guard on ``requires_grad`` of parameters: a graph traced while
    a layer was frozen is reused after the layer is unfrozen, and its backward
    never produces gradients for that layer (no error, the loss looks normal).
    Call this *before* flipping the flags at a stage boundary; the next forward
    then re-traces with the new set of trainable parameters. Returns True if the
    graphs were reset. No-op for uncompiled networks.
    """
    if not is_compiled(network):
        return False
    params = [p for n, p in network.named_parameters() if name_contains in n]
    if not params:
        return False
    currently_trainable = any(p.requires_grad for p in params)
    if currently_trainable == bool(requires_grad):
        return False
    torch.compiler.reset()
    return True
