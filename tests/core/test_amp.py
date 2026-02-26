"""Tests for automatic mixed precision (AMP) and gradient accumulation."""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dingo.core.posterior_models.base_model import train_epoch


def _make_mock_pm(device="cpu"):
    """Create a minimal mock posterior model for testing train_epoch."""
    net = nn.Linear(4, 1)
    net.to(device)

    pm = types.SimpleNamespace()
    pm.network = net
    pm.device = torch.device(device)
    pm.rank = None
    pm.epoch = 1
    pm.optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    pm.scheduler = None
    pm.scheduler_kwargs = None

    def loss_fn(theta, *context):
        return net(theta).mean()

    pm.loss = loss_fn
    return pm


def _make_dataloader(n_samples=16, batch_size=8, input_dim=4):
    theta = torch.randn(n_samples, input_dim)
    return DataLoader(TensorDataset(theta), batch_size=batch_size)


class TestAmpImports:
    def test_gradscaler_importable(self):
        from torch.amp import GradScaler

        assert GradScaler is not None

    def test_autocast_importable(self):
        from torch.amp import autocast

        assert autocast is not None

    def test_base_model_uses_torch_amp(self):
        import inspect

        from dingo.core.posterior_models import base_model

        source = inspect.getsource(base_model)
        assert "from torch.amp import" in source
        assert "from torch.cuda.amp import" not in source


class TestTrainEpochWithoutAmp:
    def test_returns_loss_and_iteration(self):
        pm = _make_mock_pm()
        dl = _make_dataloader()
        avg_loss, n_iter = train_epoch(pm, dl, automatic_mixed_precision=False)
        assert isinstance(avg_loss, float)
        assert n_iter > 0

    def test_updates_parameters(self):
        pm = _make_mock_pm()
        dl = _make_dataloader()
        params_before = [p.clone() for p in pm.network.parameters()]
        train_epoch(pm, dl, automatic_mixed_precision=False)
        params_after = list(pm.network.parameters())
        assert any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        ), "Parameters should be updated after training."

    def test_gradient_accumulation(self):
        """With 16 samples, batch=8, accum=2 → 2 batches → 1 optimizer step."""
        pm = _make_mock_pm()
        dl = _make_dataloader(n_samples=16, batch_size=8)
        _, n_iter = train_epoch(
            pm,
            dl,
            gradient_updates_per_optimizer_step=2,
            automatic_mixed_precision=False,
        )
        assert n_iter == 1

    def test_gradient_accumulation_no_step_on_incomplete_tail(self):
        """3 batches with accum=2 → only 1 complete step (last batch dropped)."""
        pm = _make_mock_pm()
        dl = _make_dataloader(n_samples=24, batch_size=8)  # 3 batches
        _, n_iter = train_epoch(
            pm,
            dl,
            gradient_updates_per_optimizer_step=2,
            automatic_mixed_precision=False,
        )
        assert n_iter == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTrainEpochWithAmp:
    def test_amp_returns_loss(self):
        pm = _make_mock_pm(device="cuda")
        dl = _make_dataloader()
        avg_loss, n_iter = train_epoch(pm, dl, automatic_mixed_precision=True)
        assert isinstance(avg_loss, float)
        assert n_iter > 0

    def test_amp_loss_is_finite(self):
        pm = _make_mock_pm(device="cuda")
        dl = _make_dataloader()
        avg_loss, _ = train_epoch(pm, dl, automatic_mixed_precision=True)
        assert torch.isfinite(torch.tensor(avg_loss))

    def test_amp_with_gradient_accumulation(self):
        pm = _make_mock_pm(device="cuda")
        dl = _make_dataloader(n_samples=16, batch_size=8)
        _, n_iter = train_epoch(
            pm,
            dl,
            gradient_updates_per_optimizer_step=2,
            automatic_mixed_precision=True,
        )
        assert n_iter == 1
