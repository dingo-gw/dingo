"""Tests for the optional ``torch.compile`` support (``local.torch_compile``)."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dingo.core.nn.compile_utils import (
    compile_network,
    eager_mode,
    is_compiled,
    reset_graphs_if_requires_grad_changes,
)
from dingo.core.nn.nsf import create_nsf_model
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.utils.torchutils import get_ddp_module, unwrap_network

_TINY_NSF_KWARGS = {
    "input_dim": 2,
    "context_dim": 4,
    "num_flow_steps": 2,
    "base_transform_kwargs": {
        "hidden_dim": 8,
        "num_transform_blocks": 1,
        "activation": "elu",
        "dropout_probability": 0.0,
        "norm": None,
        "num_bins": 4,
        "base_transform_type": "rq-coupling",
    },
}

_TINY_FLOW_METADATA = {
    "train_settings": {
        "model": {
            "posterior_model_type": "normalizing_flow",
            "posterior_kwargs": _TINY_NSF_KWARGS,
        }
    }
}


@pytest.fixture
def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def single_process_group(free_port):
    """A one-rank gloo process group, so DDP can be constructed in-process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    dist.init_process_group(
        backend="gloo", rank=0, world_size=1, timeout=timedelta(seconds=60)
    )
    yield
    dist.destroy_process_group()


def _spline_is_compile_friendly() -> bool:
    """True if the installed glasflow has the static-shape RQ spline (marked by the
    check_domain kwarg of the compile-friendly-rqs fork); released glasflow selects the
    spline tails with mask indexing, which breaks the compiled graph."""
    import inspect

    from glasflow.nflows.transforms.splines import rational_quadratic as rq

    return "check_domain" in inspect.signature(rq.rational_quadratic_spline).parameters


class TestUnwrapNetwork:
    def test_plain_module_is_returned_unchanged(self):
        net = nn.Linear(2, 2)
        assert unwrap_network(net) is net
        assert get_ddp_module(net) is None

    def test_strips_compile_wrapper(self):
        net = nn.Linear(2, 2)
        compiled = torch.compile(net)
        assert compiled is not net
        assert unwrap_network(compiled) is net
        assert get_ddp_module(compiled) is None

    def test_strips_ddp_and_compile_wrappers(self, single_process_group):
        net = nn.Linear(2, 2)
        ddp = DDP(net)
        compiled = torch.compile(ddp)
        assert unwrap_network(ddp) is net
        assert unwrap_network(compiled) is net
        assert get_ddp_module(compiled) is ddp
        assert get_ddp_module(ddp) is ddp

    def test_save_model_strips_compile_wrapper(self, tmp_path):
        pm = NormalizingFlowPosteriorModel(metadata=_TINY_FLOW_METADATA, device="cpu")
        expected_keys = set(pm.network.state_dict().keys())
        pm.network = torch.compile(pm.network)
        path = tmp_path / "model.pt"
        pm.save_model(str(path), save_training_info=False)
        saved = torch.load(path, weights_only=False)
        assert set(saved["model_state_dict"].keys()) == expected_keys


class TestEagerMode:
    def test_compiled_module_runs_without_tracing(self):
        import torch._dynamo

        torch._dynamo.reset()
        net = torch.compile(nn.Linear(2, 2))
        x = torch.randn(3, 2)
        with eager_mode():
            out = net(x)
        assert torch.allclose(out, unwrap_network(net)(x))
        assert torch._dynamo.utils.counters["frames"]["total"] == 0
        net(x)  # outside the context the module is traced and compiled
        assert torch._dynamo.utils.counters["frames"]["total"] > 0


class TestCompileNetwork:
    def test_returns_compiled_module(self):
        net = nn.Linear(2, 2)
        compiled = compile_network(net)
        assert unwrap_network(compiled) is net

    def test_per_rank_cache_dirs(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        compile_network(nn.Linear(2, 2), rank=3, cache_dir=str(tmp_path))
        inductor = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        assert inductor.startswith(str(tmp_path))
        assert "rank3" in inductor
        assert os.environ["TRITON_CACHE_DIR"].startswith(inductor)

    @pytest.mark.skipif(
        not _spline_is_compile_friendly(),
        reason="requires the static-shape RQ spline (glasflow compile-friendly-rqs fork)",
    )
    def test_flow_compiles_without_graph_breaks(self):
        """The full NSF (coupling transforms + RQ splines) must trace as one graph."""
        torch.manual_seed(0)
        flow = create_nsf_model(**_TINY_NSF_KWARGS)
        theta = torch.randn(16, 2) * 3.0  # inside and outside the spline tails
        context = torch.randn(16, 4)
        eager = flow.log_prob(theta, context)
        compiled_log_prob = torch.compile(
            flow.log_prob, fullgraph=True, backend="aot_eager"
        )
        assert torch.allclose(compiled_log_prob(theta, context), eager, atol=1e-5)


class TestResetGraphsOnRequiresGradChange:
    """Dynamo does not guard on requires_grad of parameters: a graph traced with a
    frozen layer is reused after unfreezing and never produces its gradient."""

    @staticmethod
    def _net():
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
        return net

    @staticmethod
    def _set_rb(net, requires_grad):
        for p in net[0].parameters():  # parameters "0.weight", "0.bias" play the RB layer
            p.requires_grad_(requires_grad)

    def test_unfreeze_without_reset_gives_no_gradient(self):
        net = self._net()
        self._set_rb(net, False)
        compiled = torch.compile(net, backend="aot_eager")
        x = torch.randn(8, 4)
        compiled(x).sum().backward()
        self._set_rb(net, True)
        net.zero_grad(set_to_none=True)
        compiled(x).sum().backward()
        assert net[0].weight.grad is None  # the bug this module guards against

    def test_reset_restores_gradient_after_unfreeze(self):
        net = self._net()
        self._set_rb(net, False)
        compiled = torch.compile(net, backend="aot_eager")
        x = torch.randn(8, 4)
        compiled(x).sum().backward()
        assert reset_graphs_if_requires_grad_changes(
            compiled, name_contains="0.", requires_grad=True
        )
        self._set_rb(net, True)
        net.zero_grad(set_to_none=True)
        compiled(x).sum().backward()
        assert net[0].weight.grad is not None
        assert torch.any(net[0].weight.grad != 0)

    def test_no_reset_when_state_unchanged_or_uncompiled(self):
        net = self._net()
        self._set_rb(net, False)
        assert not reset_graphs_if_requires_grad_changes(net, "0.", requires_grad=True)
        compiled = torch.compile(net, backend="aot_eager")
        assert not reset_graphs_if_requires_grad_changes(
            compiled, "0.", requires_grad=False
        )
        assert not reset_graphs_if_requires_grad_changes(
            compiled, "no_such_layer", requires_grad=True
        )

    def test_is_compiled_sees_through_ddp(self, single_process_group):
        net = nn.Linear(2, 2)
        assert not is_compiled(net)
        assert is_compiled(torch.compile(net))
        assert is_compiled(torch.compile(DDP(net)))
        assert is_compiled(DDP(torch.compile(net)))


class TestSingleGpuCacheDir:
    def test_cache_dir_honored_without_rank(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        compile_network(nn.Linear(2, 2), cache_dir=str(tmp_path))
        inductor = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        assert inductor.startswith(str(tmp_path))
        assert "rank" not in inductor
        assert os.environ["TRITON_CACHE_DIR"].startswith(inductor)
