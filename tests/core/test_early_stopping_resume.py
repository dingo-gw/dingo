"""Early stopping must survive a resume from model_latest.pt (chained jobs)."""

import torch

import dingo.core.posterior_models.base_model as base_model
from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.utils.trainutils import EarlyStopping, RuntimeLimits


class _LinearModel(BasePosteriorModel):
    """Minimal posterior model: only what `train` and `save_model` touch."""

    def initialize_network(self):
        self.network = torch.nn.Linear(1, 1)

    def sample(self, *context, num_samples=1):
        raise NotImplementedError

    def sample_and_log_prob(self, *context, num_samples=1):
        raise NotImplementedError

    def log_prob(self, theta, *context):
        raise NotImplementedError

    def loss(self, theta, *context):
        raise NotImplementedError


def _model():
    pm = _LinearModel(device="cpu")
    pm.model_kwargs = {}
    pm.optimizer = torch.optim.SGD(pm.network.parameters(), lr=0.1)
    pm.scheduler = torch.optim.lr_scheduler.ConstantLR(pm.optimizer, factor=1.0)
    return pm


def _run(pm, train_dir, losses, early_stopping, max_epochs_per_run):
    """Train with scripted test losses (indexed by epoch) up to a run limit."""
    limits = RuntimeLimits(
        max_epochs_per_run=max_epochs_per_run,
        max_epochs_total=len(losses),
        epoch_start=pm.epoch,
        device="cpu",
    )
    pm.train(
        None, None, str(train_dir), runtime_limits=limits, early_stopping=early_stopping
    )


def _resume(train_dir):
    """Rebuild model and early stopping from model_latest.pt, as train_stages does."""
    d = torch.load(train_dir / "model_latest.pt", weights_only=False)
    pm = _model()
    pm.network.load_state_dict(d["model_state_dict"])
    pm.epoch = d["epoch"]
    pm.early_stopping_state = d["early_stopping_state"]
    es = EarlyStopping(patience=3)
    es.load_state_dict(pm.early_stopping_state)
    return pm, es


def test_resume_mid_patience_stops_at_same_epoch(tmp_path, monkeypatch):
    # Best at epoch 2; epochs 3-5 are worse, so patience 3 stops at epoch 5.
    losses = [5.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5]
    monkeypatch.setattr(base_model, "train_epoch", lambda pm, **kw: (0.0, 1))
    monkeypatch.setattr(base_model, "test_epoch", lambda pm, **kw: losses[pm.epoch - 1])

    # Uninterrupted reference run.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    pm = _model()
    _run(pm, ref_dir, losses, EarlyStopping(patience=3), max_epochs_per_run=None)
    assert pm.epoch == 5

    # Same run, interrupted after epoch 3 (mid-patience) and resumed.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pm = _model()
    _run(pm, run_dir, losses, EarlyStopping(patience=3), max_epochs_per_run=3)
    best_before = torch.load(run_dir / "best_model.pt", weights_only=False)
    assert best_before["epoch"] == 2

    pm, es = _resume(run_dir)
    assert es.counter == 1 and pm.early_stopping_state["best_epoch"] == 2
    _run(pm, run_dir, losses, es, max_epochs_per_run=None)

    assert pm.epoch == 5 and es.early_stop
    assert pm.early_stopping_state["best_epoch"] == 2
    # The first epoch after the resume (worse than epoch 2) must not overwrite it.
    assert torch.load(run_dir / "best_model.pt", weights_only=False)["epoch"] == 2

    # Resuming a run that already stopped trains no further.
    pm, es = _resume(run_dir)
    _run(pm, run_dir, losses, es, max_epochs_per_run=None)
    assert pm.epoch == 5


def test_early_stopping_state_dict_round_trip():
    es = EarlyStopping(patience=2)
    es(1.0)
    es(2.0)
    restored = EarlyStopping(patience=2)
    restored.load_state_dict(es.state_dict())
    assert restored(3.0) is False and restored.early_stop
