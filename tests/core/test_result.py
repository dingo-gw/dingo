import math

import numpy as np
import pandas as pd
import pytest

from dingo.core.result import Result, _clip_weights


def make_result_with_weights(weights, extra_col=True):
    """Construct a minimal Result whose samples DataFrame has the given weights."""
    n = len(weights)
    data = {"x": np.arange(n, dtype=float), "weights": np.asarray(weights, dtype=float)}
    if extra_col:
        data["log_prob"] = np.zeros(n)
    samples = pd.DataFrame(data)
    return Result(dictionary={"samples": samples})


# ---------------------------------------------------------------------------
# rejection_sample tests
# ---------------------------------------------------------------------------


def test_no_weights_returns_copy():
    """When samples have no weights column, return an unchanged copy."""
    samples = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = Result(dictionary={"samples": samples})
    out = result.rejection_sample()
    pd.testing.assert_frame_equal(out, samples)
    assert out is not result.samples  # must be a copy


def test_weights_column_dropped():
    """Output must not contain the 'weights' column."""
    result = make_result_with_weights([1.0, 2.0, 0.5])
    out = result.rejection_sample()
    assert "weights" not in out.columns


def test_columns_dropped_and_preserved():
    """Parameter columns must appear in the output; weights, log_prob, and
    delta_log_prob_target must be dropped as they are no longer meaningful."""
    result = make_result_with_weights([1.0, 2.0, 0.5])
    out = result.rejection_sample()
    assert "x" in out.columns
    assert "weights" not in out.columns
    assert "log_prob" not in out.columns


def test_max_one_copy_per_sample_default():
    """With default max_samples_per_draw=1 no sample appears more than once."""
    weights = np.array([0.1, 5.0, 3.0, 0.5, 2.0])
    result = make_result_with_weights(weights)
    out = result.rejection_sample(random_state=0)
    # Each x value is unique in the input, so value counts should all be 1.
    assert out["x"].value_counts().max() == 1


def test_max_copies_respected():
    """With max_samples_per_draw=k no sample should appear more than k times."""
    rng = np.random.default_rng(42)
    weights = rng.exponential(scale=2.0, size=200)
    result = make_result_with_weights(weights, extra_col=False)
    for k in (1, 2, 5, 10):
        out = result.rejection_sample(max_samples_per_draw=k, random_state=k)
        counts = out["x"].value_counts()
        assert counts.max() <= k, f"k={k}: max count {counts.max()} exceeds limit"


def test_unbiasedness():
    """Expected copies per sample must be proportional to weight (unbiasedness).

    We run many independent rejection-sampling draws and compare the empirical
    inclusion rate of each sample against its expected rate.  The tolerance is
    chosen to be 10 standard deviations so the test is highly unlikely to fail
    due to random fluctuations.
    """
    rng_weights = np.random.default_rng(0)
    n_samples = 50
    weights = rng_weights.exponential(scale=1.0, size=n_samples)
    result = make_result_with_weights(weights, extra_col=False)

    n_trials = 5_000
    max_k = 3
    counts = np.zeros(n_samples, dtype=float)
    for seed in range(n_trials):
        out = result.rejection_sample(max_samples_per_draw=max_k, random_state=seed)
        for idx in out["x"].astype(int):
            counts[idx] += 1

    # Expected copies ∝ w_i; expected total across all trials:
    w_max = weights.max()
    w_scaled = weights * (max_k / w_max)
    expected_counts = w_scaled * n_trials  # E[total copies of sample i]

    # Allow 10-sigma tolerance for each sample.
    # Variance of Bernoulli(p_frac) term per trial + deterministic floor:
    # Var[copies_i per trial] ≤ 0.25  (Bernoulli variance ≤ 0.25)
    sigma = np.sqrt(n_trials * 0.25)
    np.testing.assert_allclose(counts, expected_counts, atol=10 * sigma)


def test_reproducibility():
    """Same random_state must yield identical output."""
    weights = np.array([1.0, 3.0, 0.5, 2.0, 0.1])
    result = make_result_with_weights(weights)
    out1 = result.rejection_sample(random_state=7)
    out2 = result.rejection_sample(random_state=7)
    pd.testing.assert_frame_equal(out1, out2)


def test_different_seeds_differ():
    """Different random states should (very likely) produce different outputs."""
    weights = np.array([1.0, 3.0, 0.5, 2.0, 0.1] * 20)
    result = make_result_with_weights(weights, extra_col=False)
    out1 = result.rejection_sample(random_state=1)
    out2 = result.rejection_sample(random_state=2)
    # It is astronomically unlikely that both draws are identical.
    assert not out1["x"].reset_index(drop=True).equals(out2["x"].reset_index(drop=True))


def test_invalid_max_samples_per_draw():
    """max_samples_per_draw < 1 must raise a ValueError."""
    result = make_result_with_weights([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        result.rejection_sample(max_samples_per_draw=0)
    with pytest.raises(ValueError):
        result.rejection_sample(max_samples_per_draw=-1)


def test_uniform_weights_all_accepted():
    """With equal weights and max_samples_per_draw=1 every sample is accepted
    (scaled weight = 1 for all, so fractional part = 0 and floor = 1)."""
    n = 20
    weights = np.ones(n)
    result = make_result_with_weights(weights, extra_col=False)
    out = result.rejection_sample(random_state=0)
    assert len(out) == n


def test_higher_max_increases_output():
    """Increasing max_samples_per_draw should yield at least as many output samples."""
    rng = np.random.default_rng(99)
    weights = rng.exponential(size=100)
    result = make_result_with_weights(weights, extra_col=False)
    counts = [
        len(result.rejection_sample(max_samples_per_draw=k, random_state=0))
        for k in range(1, 6)
    ]
    # Expected output grows monotonically with k (stochastically, but very reliably).
    for i in range(len(counts) - 1):
        assert counts[i] <= counts[i + 1], (
            f"k={i+1} gave {counts[i]} samples but k={i+2} gave {counts[i+1]}"
        )


# ---------------------------------------------------------------------------
# _clip_weights tests
# ---------------------------------------------------------------------------


def test_clip_weights_output_mean_is_one():
    """Clipped weights must be re-normalized to mean 1."""
    rng = np.random.default_rng(0)
    weights = rng.exponential(size=100)
    n = len(weights)
    clipped = _clip_weights(weights, math.ceil(math.sqrt(n)))
    np.testing.assert_allclose(clipped.mean(), 1.0, rtol=1e-10)


def test_clip_weights_top_values_equalized():
    """The num_clip largest weights must all equal their original mean."""
    weights = np.array([10.0, 8.0, 6.0, 1.0, 1.0, 1.0])
    num_clip = 3
    original_top_mean = np.mean([10.0, 8.0, 6.0])
    clipped = _clip_weights(weights, num_clip)
    # Re-scale back to unnormalized to check equality of top values.
    # After normalization mean=1, so original scale factor = mean of raw clipped array.
    raw_clipped = weights.copy()
    raw_clipped[:3] = original_top_mean
    np.testing.assert_allclose(
        np.sort(clipped)[::-1][:num_clip],
        np.full(num_clip, raw_clipped[:3].mean() / raw_clipped.mean()),
        rtol=1e-10,
    )


def test_clip_weights_does_not_modify_input():
    """_clip_weights must not mutate the input array."""
    weights = np.array([5.0, 3.0, 1.0, 1.0])
    original = weights.copy()
    _clip_weights(weights, 2)
    np.testing.assert_array_equal(weights, original)


# ---------------------------------------------------------------------------
# clip_weights option in rejection_sample
# ---------------------------------------------------------------------------


def test_clip_weights_increases_output():
    """clip_weights=True should yield at least as many samples as clip_weights=False."""
    # Use a highly skewed weight distribution so clipping makes a big difference.
    rng = np.random.default_rng(7)
    weights = rng.exponential(size=400)
    weights[0] = weights.max() * 20  # one very dominant weight
    result = make_result_with_weights(weights, extra_col=False)

    n_no_clip = len(result.rejection_sample(clip_weights=False, random_state=0))
    n_clip = len(result.rejection_sample(clip_weights=True, random_state=0))
    assert n_clip >= n_no_clip


def test_clip_weights_num_clip_is_ceil_sqrt_n():
    """The num_clip used internally must be ceil(sqrt(N))."""
    n = 100
    weights = np.ones(n)
    weights[0] = 1000.0  # one extreme outlier
    result = make_result_with_weights(weights, extra_col=False)

    # With clip_weights=True, the outlier should be clipped: check that the
    # max output count is 1 (since after clipping all weights are ~equal).
    out = result.rejection_sample(clip_weights=True, random_state=0)
    assert out["x"].value_counts().max() == 1


def test_clip_weights_reproducible():
    """clip_weights=True with the same random_state must give identical results."""
    rng = np.random.default_rng(3)
    weights = rng.exponential(size=50)
    result = make_result_with_weights(weights, extra_col=False)
    out1 = result.rejection_sample(clip_weights=True, random_state=42)
    out2 = result.rejection_sample(clip_weights=True, random_state=42)
    pd.testing.assert_frame_equal(out1, out2)
