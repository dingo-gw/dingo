"""Tests for automatic mixed precision (AMP) and gradient accumulation."""

import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dingo.core.posterior_models import base_model
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
        try:
            from torch.amp import GradScaler
        except ImportError:
            from torch.cuda.amp import GradScaler
        assert GradScaler is not None

    def test_autocast_importable(self):
        from torch.amp import autocast

        assert autocast is not None

    def test_base_model_uses_torch_amp(self):
        import inspect

        from dingo.core.posterior_models import base_model

        source = inspect.getsource(base_model)
        assert "from torch.amp import" in source


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

    @pytest.mark.parametrize("log_grad_norm", [False, True])
    def test_grad_norm_logged_only_when_requested(self, log_grad_norm, capsys):
        pm = _make_mock_pm()
        train_epoch(pm, _make_dataloader(), log_interval=1, log_grad_norm=log_grad_norm)
        assert ("Grad norm" in capsys.readouterr().out) == log_grad_norm

    def test_amp_unscales_explicitly_on_every_step(self):
        # Without the explicit unscale, a fused optimizer unscales in float64 (slow);
        # independent of the gradient-norm logging.
        pm = _make_mock_pm()
        scaler = torch.amp.GradScaler("cpu")
        calls = []
        unscale = scaler.unscale_
        scaler.unscale_ = lambda opt: (calls.append(1), unscale(opt))[1]
        _, n_iter = train_epoch(
            pm, _make_dataloader(), automatic_mixed_precision=True, scaler=scaler
        )
        assert len(calls) == n_iter > 0

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

    def test_gradient_accumulation_matches_full_batch(self):
        """Accumulating 2 x 8 samples must update like a single batch of 16."""
        torch.manual_seed(0)
        theta = torch.randn(16, 4)
        pm_full = _make_mock_pm()
        pm_accum = _make_mock_pm()
        pm_accum.network.load_state_dict(pm_full.network.state_dict())

        train_epoch(pm_full, DataLoader(TensorDataset(theta), batch_size=16))
        train_epoch(
            pm_accum,
            DataLoader(TensorDataset(theta), batch_size=8),
            gradient_updates_per_optimizer_step=2,
        )
        for p_full, p_accum in zip(
            pm_full.network.parameters(), pm_accum.network.parameters()
        ):
            torch.testing.assert_close(p_accum, p_full)

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


class TestLogInterval:
    def test_epoch_average_independent_of_log_interval(self, capsys):
        """The epoch-average loss and the trained weights do not depend on how often
        the loss is read from the device; the partial last window is printed too."""
        torch.manual_seed(0)
        theta = torch.randn(40, 4)  # 5 batches of 8; accumulate 2 -> 2 steps + tail
        results = {}
        for log_interval in (1, 3):
            torch.manual_seed(1)
            pm = _make_mock_pm()
            avg, n_iter = train_epoch(
                pm,
                DataLoader(TensorDataset(theta), batch_size=8),
                gradient_updates_per_optimizer_step=2,
                log_interval=log_interval,
            )
            # base_model.test_epoch: importing the name would make pytest collect it.
            test_avg = base_model.test_epoch(
                pm,
                DataLoader(TensorDataset(theta), batch_size=8),
                log_interval=log_interval,
            )
            lines = [
                l for l in capsys.readouterr().out.splitlines() if l.startswith("Train")
            ]
            assert n_iter == 2 and len(lines) == {1: 2, 3: 1}[log_interval]
            results[log_interval] = (avg, test_avg, list(pm.network.parameters()))
        assert results[1][0] == results[3][0]  # same float64 sums -> identical
        assert results[1][1] == results[3][1]
        assert all(torch.equal(a, b) for a, b in zip(results[1][2], results[3][2]))
        assert "Time per step" in lines[-1] and "mean over 2 steps" in lines[-1]


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


class TestTestEpochWithAmp:
    def test_validation_runs_under_autocast(self):
        """
        With automatic_mixed_precision the validation forward pass must run under
        autocast (for the model's device type), like the training pass; without
        it, autocast must not be entered.
        """
        from unittest import mock

        from torch.amp import autocast

        from dingo.core.posterior_models import base_model

        pm = _make_mock_pm()
        loader = _make_dataloader()
        with mock.patch.object(base_model, "autocast", wraps=autocast) as ac:
            loss = base_model.test_epoch(pm, loader, automatic_mixed_precision=True)
            assert ac.call_count == 1
            assert ac.call_args.args[0] == "cpu"
            assert torch.isfinite(torch.tensor(loss))
        with mock.patch.object(base_model, "autocast", wraps=autocast) as ac:
            base_model.test_epoch(pm, loader)
            ac.assert_not_called()
