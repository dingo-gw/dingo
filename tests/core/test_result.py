import numpy as np
import pandas as pd
import pytest

from dingo.core.result import Result


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


def test_other_columns_preserved():
    """Non-weight columns (e.g., log_prob) must appear in the output."""
    result = make_result_with_weights([1.0, 2.0, 0.5])
    out = result.rejection_sample()
    assert "x" in out.columns
    assert "log_prob" in out.columns


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
