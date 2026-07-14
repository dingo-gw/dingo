import numpy as np

from dingo.core.utils.gnpeutils import IterationTracker


def test_pvalue_min_is_minus_inf_before_two_updates():
    tracker = IterationTracker()
    assert tracker.pvalue_min == -np.inf  # no ks_result yet

    tracker.update({"a": np.random.default_rng(0).normal(size=100)})
    # After a single update there is still no KS comparison.
    assert tracker.pvalue_min == -np.inf


def test_first_update_stores_data_with_leading_axis():
    tracker = IterationTracker()
    tracker.update({"a": np.zeros(50), "b": np.ones(50)})
    assert tracker.data["a"].shape == (1, 50)
    assert tracker.data["b"].shape == (1, 50)


def test_second_update_computes_ks_pvalue():
    rng = np.random.default_rng(0)
    tracker = IterationTracker()
    tracker.update({"a": rng.normal(size=200)})
    tracker.update({"a": rng.normal(size=200)})
    # Two samples from the same distribution -> a finite, valid p-value.
    assert tracker.ks_result is not None
    assert np.isfinite(tracker.pvalue_min)
    assert 0.0 <= tracker.pvalue_min <= 1.0


def test_store_data_true_accumulates_rows():
    rng = np.random.default_rng(0)
    tracker = IterationTracker(store_data=True)
    for _ in range(3):
        tracker.update({"a": rng.normal(size=20)})
    # store_data=True keeps every iteration along axis 0.
    assert tracker.data["a"].shape == (3, 20)


def test_store_data_false_keeps_only_latest():
    rng = np.random.default_rng(0)
    tracker = IterationTracker(store_data=False)
    for _ in range(3):
        tracker.update({"a": rng.normal(size=20)})
    # store_data=False replaces the stored data each iteration.
    assert tracker.data["a"].shape == (1, 20)
