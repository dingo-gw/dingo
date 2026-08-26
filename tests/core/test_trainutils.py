import math
import os

import pytest

from dingo.core.utils.trainutils import (
    AvgTracker,
    EarlyStopping,
    LossInfo,
    RuntimeLimits,
    write_history,
)


# ---------------------------------------------------------------------------
# AvgTracker
# ---------------------------------------------------------------------------


def test_avg_tracker_empty_returns_nan():
    tracker = AvgTracker()
    assert math.isnan(tracker.get_avg())


def test_avg_tracker_weighted_average_and_last_value():
    tracker = AvgTracker()
    tracker.update(2.0)
    tracker.update(4.0)
    assert tracker.get_avg() == 3.0
    assert tracker.x == 4.0

    # With explicit counts the average is sum / total-count.
    tracker = AvgTracker()
    tracker.update(10.0, n=2)
    tracker.update(2.0, n=3)
    assert tracker.get_avg() == 12.0 / 5


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


def test_early_stopping_first_call_sets_best_and_returns_true():
    es = EarlyStopping(patience=3)
    assert es(1.0) is True
    assert es.best_score == -1.0
    assert es.counter == 0
    assert es.early_stop is False


def test_early_stopping_improving_loss_keeps_counter_zero():
    es = EarlyStopping(patience=3)
    es(5.0)
    for loss in (4.0, 3.0, 2.0):
        assert es(loss) is True
    assert es.counter == 0
    assert es.early_stop is False


def test_early_stopping_triggers_after_patience_non_improving():
    es = EarlyStopping(patience=3)
    es(1.0)  # best
    # Three consecutive non-improving losses reach the patience limit.
    assert es(2.0) is False and es.counter == 1
    assert es(2.0) is False and es.counter == 2
    assert es(2.0) is False and es.counter == 3
    assert es.early_stop is True


def test_early_stopping_delta_makes_small_improvement_count_as_non_improving():
    # An improvement must exceed `delta` to reset the counter.
    es = EarlyStopping(patience=5, delta=1.0)
    es(10.0)  # best_score = -10
    # New loss 9.5 -> score -9.5, which is < best_score + delta = -9.0, so non-improving.
    assert es(9.5) is False
    assert es.counter == 1


def test_early_stopping_invalid_metric_raises():
    with pytest.raises(ValueError, match="training.*validation|validation.*training"):
        EarlyStopping(metric="not_a_metric")


# ---------------------------------------------------------------------------
# RuntimeLimits
# ---------------------------------------------------------------------------


def test_runtime_limits_none_never_exceeded():
    limits = RuntimeLimits()
    assert limits.limits_exceeded(epoch=10_000) is False


def test_runtime_limits_total_epochs():
    limits = RuntimeLimits(max_epochs_total=10)
    assert limits.limits_exceeded(epoch=9) is False
    assert limits.limits_exceeded(epoch=10) is True


def test_runtime_limits_epochs_per_run():
    limits = RuntimeLimits(max_epochs_per_run=5, epoch_start=3)
    assert limits.limits_exceeded(epoch=7) is False  # 7 - 3 = 4 < 5
    assert limits.limits_exceeded(epoch=8) is True  # 8 - 3 = 5 >= 5


def test_runtime_limits_epochs_per_run_requires_epoch():
    limits = RuntimeLimits(max_epochs_per_run=5, epoch_start=0)
    with pytest.raises(ValueError, match="epoch required"):
        limits.limits_exceeded(epoch=None)


def test_runtime_limits_per_run_requires_epoch_start():
    with pytest.raises(ValueError, match="epoch_start required"):
        RuntimeLimits(max_epochs_per_run=5)


def test_runtime_limits_time_limit_zero_is_exceeded_immediately():
    # Any elapsed time >= 0 trips a zero time limit.
    limits = RuntimeLimits(max_time_per_run=0.0)
    assert limits.limits_exceeded(epoch=0) is True


def test_local_limits_ignore_total_epoch_limit():
    # local_limits_exceeded honours per-run/time limits but not the total epoch limit.
    limits = RuntimeLimits(max_epochs_total=5)
    assert limits.local_limits_exceeded(epoch=100) is False

    limits = RuntimeLimits(max_epochs_per_run=2, epoch_start=0)
    assert limits.local_limits_exceeded(epoch=1) is False
    assert limits.local_limits_exceeded(epoch=2) is True


# ---------------------------------------------------------------------------
# LossInfo
# ---------------------------------------------------------------------------


def test_loss_info_weighted_average_across_batches():
    info = LossInfo(epoch=1, len_dataset=100, batch_size=10)
    info.update(2.0, n=4)
    info.update(3.0, n=2)
    # Weighted: (2*4 + 3*2) / (4 + 2) = 14 / 6.
    assert info.get_avg() == pytest.approx(14.0 / 6)
    assert info.loss == 3.0


# ---------------------------------------------------------------------------
# write_history
# ---------------------------------------------------------------------------


def test_write_history_appends_rows(tmp_path):
    log_dir = str(tmp_path)
    write_history(log_dir, 1, -1.0, -0.5, [0.001])
    write_history(log_dir, 2, -2.0, -1.5, [0.0005])

    history_file = os.path.join(log_dir, "history.txt")
    with open(history_file) as f:
        rows = [line.strip().split("\t") for line in f if line.strip()]

    assert len(rows) == 2
    assert rows[0] == ["1", "-1.0", "-0.5", "0.001"]
    assert rows[1] == ["2", "-2.0", "-1.5", "0.0005"]


def test_write_history_refuses_to_overwrite_on_first_epoch(tmp_path):
    log_dir = str(tmp_path)
    write_history(log_dir, 1, -1.0, -0.5, [0.001])
    # Writing epoch 1 again would clobber the existing file.
    with pytest.raises(AssertionError):
        write_history(log_dir, 1, -1.0, -0.5, [0.001])
