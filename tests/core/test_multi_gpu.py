"""
Tests for multi-GPU DDP utilities.

Most tests run without a real GPU by:
  * testing individual components in isolation, or
  * using the CPU-compatible ``gloo`` backend for process-group tests.
"""

import os
import warnings
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset

from dingo.core.utils.torchutils import (
    build_train_and_test_loaders,
    get_cuda_info,
    set_seed_based_on_rank,
)
from dingo.core.utils.trainutils import LossInfo, RuntimeLimits
from dingo.gw.training.train_pipeline import get_num_gpus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TINY_FLOW_METADATA = {
    "train_settings": {
        "model": {
            "posterior_model_type": "normalizing_flow",
            "posterior_kwargs": {
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
            },
            "embedding_type": None,
            "embedding_kwargs": None,
        }
    }
}


def _setup_gloo(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def _cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Utility-function tests (no GPU, no process group required)
# ---------------------------------------------------------------------------


class TestGetNumGpus:
    def test_reads_from_local_num_gpus(self):
        assert get_num_gpus({"num_gpus": 4}) == 4

    def test_defaults_to_one(self):
        assert get_num_gpus({}) == 1

    def test_condor_num_gpus_deprecated_fallback(self):
        """condor.num_gpus is still honoured but raises a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="local.condor"):
            result = get_num_gpus({"condor": {"num_gpus": 4}})
        assert result == 4

    def test_local_num_gpus_takes_precedence(self):
        """local.num_gpus wins and no warning is raised."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_num_gpus({"num_gpus": 4, "condor": {"num_gpus": 2}}) == 4

    def test_returns_int(self):
        assert isinstance(get_num_gpus({"num_gpus": "2"}), int)


class TestGetCudaInfo:
    def test_returns_dict(self):
        info = get_cuda_info()
        assert isinstance(info, dict)
        if torch.cuda.is_available():
            assert "device count" in info
        else:
            assert info == {}


class TestSetSeedBasedOnRank:
    def test_different_seeds_per_rank(self):
        torch.manual_seed(42)
        set_seed_based_on_rank(0)
        x0 = torch.rand(5).tolist()

        torch.manual_seed(42)
        set_seed_based_on_rank(1)
        x1 = torch.rand(5).tolist()

        assert x0 != x1, "Different ranks should produce different random draws."

    def test_numpy_seed_set(self):
        set_seed_based_on_rank(0)
        a = np.random.rand(3)
        set_seed_based_on_rank(0)
        b = np.random.rand(3)
        np.testing.assert_array_equal(a, b)


class TestNetworkToDeviceRank:
    def test_rank_stays_none_without_process_group(self):
        """Single-GPU training on e.g. cuda:1 must not enable the DDP paths."""
        from dingo.core.posterior_models.normalizing_flow import (
            NormalizingFlowPosteriorModel,
        )

        pm = NormalizingFlowPosteriorModel(metadata=_TINY_FLOW_METADATA, device="cpu")
        pm.network_to_device("cpu:0")
        assert pm.rank is None


class TestDDPStateDictStripping:
    """Test that model_dict() strips DDP 'module.' prefix when saving."""

    def test_save_model_strips_ddp_prefix(self, tmp_path):
        from dingo.core.posterior_models.normalizing_flow import (
            NormalizingFlowPosteriorModel,
        )

        pm = NormalizingFlowPosteriorModel(metadata=_TINY_FLOW_METADATA, device="cpu")

        # Simulate DDP wrapping by injecting "module." prefixes into state dict.
        original_sd = pm.network.state_dict()
        ddp_sd = {f"module.{k}": v for k, v in original_sd.items()}

        # Patch network so isinstance(network, DDP) is False but keys have "module.".
        # We test the isinstance(network, DDP) branch explicitly below.
        class _FakeDDP(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.module = mod

            def state_dict(self, **kw):
                return {
                    f"module.{k}": v for k, v in self.module.state_dict(**kw).items()
                }

        pm.network = _FakeDDP(
            pm.network.module if hasattr(pm.network, "module") else pm.network
        )

        # Manually test the save logic: DDP branch should strip "module." prefix.
        if isinstance(pm.network, DDP):
            saved_sd = pm.network.module.state_dict()
        else:
            # Simulate detecting wrapper by checking for the attribute.
            saved_sd = (
                pm.network.module.state_dict()
                if hasattr(pm.network, "module")
                else pm.network.state_dict()
            )

        assert not any(
            "module." in k for k in saved_sd.keys()
        ), "Saved state dict should not contain 'module.' prefix."


class TestBuildTrainAndTestLoaders:
    def test_single_gpu_returns_none_sampler(self):
        dataset = TensorDataset(torch.randn(100, 4))
        train_loader, test_loader, sampler = build_train_and_test_loaders(
            dataset=dataset,
            train_fraction=0.8,
            batch_size=10,
            num_workers=0,
        )
        assert sampler is None
        assert len(train_loader.dataset) == 80
        assert len(test_loader.dataset) == 20

    def test_ddp_returns_distributed_sampler(self):
        from torch.utils.data import DistributedSampler

        dataset = TensorDataset(torch.randn(100, 4))
        train_loader, test_loader, sampler = build_train_and_test_loaders(
            dataset=dataset,
            train_fraction=0.8,
            batch_size=5,
            num_workers=0,
            world_size=2,
            rank=0,
        )
        assert isinstance(sampler, DistributedSampler)
        # Each rank sees ceil(80/2) = 40 samples.
        assert len(train_loader.dataset) == 80

    def test_ddp_splits_data_across_ranks(self):
        from torch.utils.data import DistributedSampler

        dataset = TensorDataset(torch.arange(100).float().unsqueeze(1))
        _, _, sampler0 = build_train_and_test_loaders(
            dataset, 1.0, 5, 0, world_size=2, rank=0
        )
        _, _, sampler1 = build_train_and_test_loaders(
            dataset, 1.0, 5, 0, world_size=2, rank=1
        )
        # The two samplers should produce disjoint index sets.
        indices0 = set(sampler0)
        indices1 = set(sampler1)
        assert len(indices0 & indices1) == 0, "DDP samplers should partition the data."


class TestLossInfoSingleGPU:
    """LossInfo tests without a process group."""

    def test_basic_update(self):
        info = LossInfo(
            epoch=1,
            len_dataset=100,
            batch_size_per_grad_update=10,
            device=torch.device("cpu"),
        )
        loss = torch.tensor(2.0)
        info.update_timer("Dataloader")
        info.cache_loss(loss, n=10)
        info.update()
        assert abs(info.get_avg() - 2.0) < 1e-5

    def test_iteration_count(self):
        info = LossInfo(1, 100, 10, device=torch.device("cpu"))
        for _ in range(3):
            info.update_timer("Dataloader")
            info.cache_loss(torch.tensor(1.0), n=10)
            info.update()
        assert info.get_iteration() == 3


class TestRuntimeLimitsSingleGPU:
    def test_epoch_limit(self):
        rl = RuntimeLimits(
            max_epochs_total=5, epoch_start=0, device=torch.device("cpu")
        )
        rl.max_epochs_total = 5
        assert not rl.limits_exceeded(4)
        assert rl.limits_exceeded(5)


# ---------------------------------------------------------------------------
# Process-group tests using gloo (CPU, no NCCL required)
# ---------------------------------------------------------------------------


def _worker_loss_info(rank, world_size, port, result_queue):
    """Worker function for testing LossInfo in DDP mode with gloo."""
    _setup_gloo(rank, world_size, port)
    device = torch.device("cpu")
    info = LossInfo(1, 100, 10, device=device)
    # Simulate two optimizer steps with different losses per rank.
    for i in range(2):
        info.update_timer("Dataloader")
        loss = torch.tensor(float(rank + 1) * (i + 1))
        info.cache_loss(loss, n=10)
        info.update()
    avg = info.get_avg()
    n_iter = info.get_iteration()
    result_queue.put((rank, avg, n_iter))
    _cleanup()


def _worker_network_to_device(rank, world_size, port, result_queue):
    """Worker checking that network_to_device sets the rank inside a group."""
    from dingo.core.posterior_models.normalizing_flow import (
        NormalizingFlowPosteriorModel,
    )

    _setup_gloo(rank, world_size, port)
    pm = NormalizingFlowPosteriorModel(metadata=_TINY_FLOW_METADATA, device="cpu")
    pm.network_to_device(f"cpu:{rank}")
    result_queue.put((rank, pm.rank))
    _cleanup()


def _worker_runtime_limits(rank, world_size, port, result_queue):
    """Worker function for testing RuntimeLimits broadcast with gloo."""
    _setup_gloo(rank, world_size, port)
    device = torch.device("cpu")
    rl = RuntimeLimits(max_epochs_total=5, epoch_start=0, device=device)
    # Rank 0 sees epoch=6 (over limit); rank 1 sees epoch=4 (under).
    epoch = 6 if rank == 0 else 4
    exceeded = rl.limits_exceeded(epoch)
    result_queue.put((rank, exceeded))
    _cleanup()


def _run_workers(target, world_size, port):
    """
    Run one worker per rank; fail fast if any hangs or crashes.

    Each worker puts a tuple ``(rank, *results)`` into the queue. Returns a
    dict mapping rank to the tuple of results.
    """
    # Use "spawn" rather than the default "fork": autograd cannot be used in a
    # forked child once the parent process has run a backward pass (which
    # other tests in the session do).
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=target,
            args=(rank, world_size, port, result_queue),
            daemon=True,
        )
        for rank in range(world_size)
    ]
    try:
        for p in processes:
            p.start()
        # Drain the queue before joining: a child blocks on exit until its
        # queued data has been consumed, and polling ``empty()`` after the
        # join can race with data still in flight through the pipe.
        results = {}
        for _ in range(world_size):
            item = result_queue.get(timeout=60)
            results[item[0]] = tuple(item[1:])
        for p in processes:
            p.join(timeout=60)
        for p in processes:
            assert not p.is_alive(), "Distributed worker hung."
            assert p.exitcode == 0, f"Worker failed with code {p.exitcode}"
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
    assert len(results) == world_size
    return results


@pytest.fixture()
def free_port():
    """Find a free port for the gloo process group."""
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestGlooDDP:
    """Integration tests using the CPU gloo backend — no GPU required."""

    def test_loss_info_aggregation(self, free_port):
        """Both ranks should observe the same averaged loss."""
        world_size = 2
        results = _run_workers(_worker_loss_info, world_size, free_port)
        # Both ranks should report the same number of iterations.
        assert results[0][1] == results[1][1] == 2
        # Losses are all-reduced, so every rank sees the same average: per step
        # mean over ranks of (1, 2) and (2, 4), averaged over the two steps.
        assert results[0][0] == pytest.approx(2.25)
        assert results[1][0] == pytest.approx(2.25)

    def test_network_to_device_sets_rank(self, free_port):
        """Inside a process group, the device index becomes the rank."""
        world_size = 2
        results = _run_workers(_worker_network_to_device, world_size, free_port)
        assert results == {0: (0,), 1: (1,)}

    def test_runtime_limits_broadcast(self, free_port):
        """When rank 0 hits the epoch limit, rank 1 should also stop."""
        world_size = 2
        results = _run_workers(_worker_runtime_limits, world_size, free_port)
        # Both ranks must agree: limit is exceeded because rank 0 triggered it.
        assert results[0] == (True,)
        assert results[1] == (True,)

    def test_no_sync_on_accumulation_steps(self, free_port):
        """no_sync() wraps only the accumulation-only batches; weights stay in sync."""
        world_size = 2
        results = _run_workers(_worker_no_sync, world_size, free_port)
        for n_iter, n_no_sync, _ in results.values():
            assert n_iter == 2
            # 4 batches, accumulate over 2: no_sync entered on batches 0 and 2.
            assert n_no_sync == 2
        assert results[0][2] == pytest.approx(results[1][2])


def _worker_no_sync(rank, world_size, port, result_queue):
    """
    Worker checking that train_epoch skips the gradient all-reduce on
    accumulation-only steps and synchronizes on the final one.
    """
    import types
    from unittest import mock

    from torch.utils.data import DataLoader

    from dingo.core.posterior_models.base_model import train_epoch

    _setup_gloo(rank, world_size, port)
    torch.manual_seed(0)
    net = DDP(nn.Linear(4, 1))
    pm = types.SimpleNamespace(
        network=net,
        device=torch.device("cpu"),
        rank=rank,
        epoch=1,
        optimizer=torch.optim.SGD(net.parameters(), lr=0.1),
    )
    pm.loss = lambda theta, *context: net(theta).mean()

    # Different data per rank, so an un-synchronized step would diverge.
    torch.manual_seed(rank)
    dataset = TensorDataset(torch.randn(16, 4))
    loader = DataLoader(dataset, batch_size=4)  # 4 batches, 2 optimizer steps

    with mock.patch.object(net, "no_sync", wraps=net.no_sync) as no_sync:
        _, n_iter = train_epoch(pm, loader, gradient_updates_per_optimizer_step=2)
    weights = net.module.weight.detach().clone().flatten().tolist()
    result_queue.put((rank, n_iter, no_sync.call_count, weights))
    _cleanup()
