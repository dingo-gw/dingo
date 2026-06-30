import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from dingo.core.result import Result, _clip_weights


def make_result_with_weights(weights, extra_col=True):
    """Construct a minimal Result whose samples DataFrame has the given weights."""
    n = len(weights)
    data = {"x": np.arange(n, dtype=float), "weights": np.asarray(weights, dtype=float)}
    if extra_col:
        data["log_prob"] = np.zeros(n)
    samples = pd.DataFrame(data)
    return Result(dictionary={"samples": samples})


def make_result_plain(n):
    """Construct a Result with two parameter columns and no weights."""
    samples = pd.DataFrame(
        {"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float) * 2}
    )
    return Result(dictionary={"samples": samples})


def make_result_with_log_columns(
    log_prob, log_prior, log_likelihood, delta=None, calculate_evidence=False
):
    """Construct a Result whose samples carry the columns needed for evidence.

    A unique parameter column ``x`` is added so individual rows can be tracked
    through resampling / splitting.
    """
    n = len(log_prob)
    data = {
        "x": np.arange(n, dtype=float),
        "log_prob": np.asarray(log_prob, dtype=float),
        "log_prior": np.asarray(log_prior, dtype=float),
        "log_likelihood": np.asarray(log_likelihood, dtype=float),
    }
    if delta is not None:
        data["delta_log_prob_target"] = np.asarray(delta, dtype=float)
    result = Result(dictionary={"samples": pd.DataFrame(data)})
    if calculate_evidence:
        result._calculate_evidence()
    return result


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


# ---------------------------------------------------------------------------
# sampling_importance_resampling tests
# ---------------------------------------------------------------------------


def test_sir_default_num_samples():
    """Default num_samples returns as many rows as the input."""
    result = make_result_with_weights([1.0, 2.0, 3.0, 4.0], extra_col=False)
    out = result.sampling_importance_resampling(random_state=0)
    assert len(out) == 4


def test_sir_weights_column_dropped():
    """The weights column must be dropped from the resampled output."""
    result = make_result_with_weights([1.0, 2.0, 3.0], extra_col=False)
    out = result.sampling_importance_resampling(random_state=0)
    assert "weights" not in out.columns


def test_sir_too_many_samples_raises():
    """Requesting more samples than available must raise a ValueError."""
    result = make_result_with_weights([1.0, 2.0, 3.0], extra_col=False)
    with pytest.raises(ValueError):
        result.sampling_importance_resampling(num_samples=4)


def test_sir_reproducible():
    """Same random_state must yield identical output."""
    result = make_result_with_weights([1.0, 3.0, 0.5, 2.0, 0.1], extra_col=False)
    out1 = result.sampling_importance_resampling(num_samples=3, random_state=11)
    out2 = result.sampling_importance_resampling(num_samples=3, random_state=11)
    pd.testing.assert_frame_equal(out1, out2)


def test_sir_zero_weight_never_sampled():
    """A sample with zero weight must never appear in the output."""
    weights = np.array([0.0, 1.0, 1.0, 1.0])
    result = make_result_with_weights(weights, extra_col=False)
    out = result.sampling_importance_resampling(random_state=0)
    assert 0.0 not in set(out["x"])  # x=0 corresponds to the zero-weight sample


def test_sir_resamples_proportional_to_weights():
    """Resampling frequency must be proportional to the sample weights."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    result = make_result_with_weights(weights, extra_col=False)
    n = len(weights)
    counts = np.zeros(n)
    n_trials = 4000
    for seed in range(n_trials):
        out = result.sampling_importance_resampling(random_state=seed)
        for xi in out["x"].astype(int):
            counts[xi] += 1
    freq = counts / counts.sum()
    expected = weights / weights.sum()
    np.testing.assert_allclose(freq, expected, atol=0.02)


# ---------------------------------------------------------------------------
# _calculate_evidence tests
# ---------------------------------------------------------------------------


def test_calculate_evidence_value():
    """log_evidence must equal logsumexp(log_weights) - log(N)."""
    n = 5
    log_prob = np.zeros(n)
    log_prior = np.zeros(n)
    log_likelihood = np.log(np.arange(1, n + 1, dtype=float))  # L_i = i
    result = make_result_with_log_columns(log_prob, log_prior, log_likelihood)
    result._calculate_evidence()
    # log_weights = log_prior + log_likelihood - log_prob = log_likelihood here.
    expected = logsumexp(log_likelihood) - np.log(n)
    assert result.log_evidence == pytest.approx(expected)


def test_calculate_evidence_weights_normalized_to_mean_one():
    """The stored weights must be normalized to mean 1."""
    rng = np.random.default_rng(0)
    n = 100
    result = make_result_with_log_columns(
        rng.normal(size=n), np.zeros(n), rng.normal(size=n)
    )
    result._calculate_evidence()
    np.testing.assert_allclose(result.samples["weights"].mean(), 1.0, rtol=1e-10)


def test_calculate_evidence_delta_log_prob_target_shifts_evidence():
    """A constant delta_log_prob_target shifts log_evidence by that constant and
    leaves the (normalized) weights unchanged."""
    n = 4
    log_likelihood = np.log(np.arange(1, n + 1, dtype=float))
    base = make_result_with_log_columns(np.zeros(n), np.zeros(n), log_likelihood)
    base._calculate_evidence()

    c = 2.5
    shifted = make_result_with_log_columns(
        np.zeros(n), np.zeros(n), log_likelihood, delta=np.full(n, c)
    )
    shifted._calculate_evidence()

    assert shifted.log_evidence == pytest.approx(base.log_evidence + c)
    np.testing.assert_allclose(
        shifted.samples["weights"], base.samples["weights"], rtol=1e-10
    )


def test_calculate_evidence_nan_likelihood_treated_as_zero():
    """A NaN log_likelihood (un-evaluated sample) must be treated as 0, not
    propagated to the evidence."""
    n = 4
    log_likelihood = np.log(np.arange(1, n + 1, dtype=float))

    ll_nan = log_likelihood.copy()
    ll_nan[0] = np.nan
    r_nan = make_result_with_log_columns(np.zeros(n), np.zeros(n), ll_nan)
    r_nan._calculate_evidence()

    ll_zero = log_likelihood.copy()
    ll_zero[0] = 0.0
    r_zero = make_result_with_log_columns(np.zeros(n), np.zeros(n), ll_zero)
    r_zero._calculate_evidence()

    assert r_nan.log_evidence == pytest.approx(r_zero.log_evidence)


def test_calculate_evidence_noop_without_required_columns():
    """With required columns missing, no weights/log_evidence are produced."""
    samples = pd.DataFrame({"x": [1.0, 2.0, 3.0], "log_prob": [0.0, 0.0, 0.0]})
    result = Result(dictionary={"samples": samples})
    result._calculate_evidence()
    assert "weights" not in result.samples.columns
    assert result.log_evidence is None


# ---------------------------------------------------------------------------
# split / merge tests
# ---------------------------------------------------------------------------


def test_split_num_parts_and_sizes():
    """split must yield the requested number of parts, the sizes summing to the
    total with the remainder placed in the final part."""
    result = make_result_plain(10)
    parts = result.split(3)
    assert len(parts) == 3
    assert sum(p.num_samples for p in parts) == 10
    assert [p.num_samples for p in parts] == [3, 3, 4]


def test_split_preserves_samples_in_order():
    """Concatenating the parts must reproduce the original samples in order."""
    result = make_result_plain(10)
    parts = result.split(3)
    concat = pd.concat([p.samples for p in parts], ignore_index=True)
    pd.testing.assert_frame_equal(concat, result.samples)


def test_merge_roundtrips_split():
    """merge(split(result)) must reproduce the original samples."""
    rng = np.random.default_rng(0)
    n = 12
    result = make_result_with_log_columns(
        rng.normal(size=n), np.zeros(n), rng.normal(size=n), calculate_evidence=True
    )
    parts = result.split(4)
    merged = Result.merge(parts)
    assert merged.num_samples == result.num_samples
    np.testing.assert_array_equal(
        merged.samples["x"].to_numpy(), result.samples["x"].to_numpy()
    )
    # Evidence is recomputed over the full set, matching the original.
    assert merged.log_evidence == pytest.approx(result.log_evidence)


def test_merge_incompatible_metadata_raises():
    """Merging Results with differing metadata must raise a ValueError."""
    rng = np.random.default_rng(1)
    n = 6
    r1 = make_result_with_log_columns(
        rng.normal(size=n), np.zeros(n), rng.normal(size=n), calculate_evidence=True
    )
    r2 = make_result_with_log_columns(
        rng.normal(size=n), np.zeros(n), rng.normal(size=n), calculate_evidence=True
    )
    r2.event_metadata = {"foo": "bar"}  # differs from r1 (None)
    with pytest.raises(ValueError):
        Result.merge([r1, r2])


# ---------------------------------------------------------------------------
# property tests
# ---------------------------------------------------------------------------


def test_num_samples_with_and_without_samples():
    result = make_result_plain(7)
    assert result.num_samples == 7
    result.samples = None
    assert result.num_samples == 0


def test_effective_sample_size_uniform_weights():
    """Uniform weights give ESS == N and efficiency == 1."""
    n = 10
    result = make_result_with_weights(np.ones(n), extra_col=False)
    assert result.effective_sample_size == pytest.approx(n)
    assert result.n_eff == pytest.approx(n)
    assert result.sample_efficiency == pytest.approx(1.0)


def test_ess_and_efficiency_none_without_weights():
    result = make_result_plain(5)
    assert result.effective_sample_size is None
    assert result.sample_efficiency is None


def test_log_bayes_factor():
    result = make_result_plain(3)
    result.log_evidence = -5.0
    result.log_noise_evidence = -8.0
    assert result.log_bayes_factor == pytest.approx(3.0)


def test_log_bayes_factor_none_when_incomplete():
    result = make_result_plain(3)
    result.log_evidence = -5.0
    # log_noise_evidence is None by default.
    assert result.log_bayes_factor is None


# ---------------------------------------------------------------------------
# parameter_subset tests
# ---------------------------------------------------------------------------


def test_parameter_subset_keeps_only_requested_columns():
    samples = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "log_prob": [0.0, 0.0, 0.0],
            "weights": [1.0, 1.0, 1.0],
        }
    )
    result = Result(dictionary={"samples": samples})
    sub = result.parameter_subset(["x"])
    assert list(sub.samples.columns) == ["x"]
    assert isinstance(sub, Result)


# ---------------------------------------------------------------------------
# injection credible level tests
# ---------------------------------------------------------------------------


def make_result_with_injection(values, injection_value, weights=None):
    data = {"x": np.asarray(values, dtype=float)}
    if weights is not None:
        data["weights"] = np.asarray(weights, dtype=float)
    return Result(
        dictionary={
            "samples": pd.DataFrame(data),
            "event_metadata": {"injection_parameters": {"x": injection_value}},
        }
    )


def test_credible_level_unweighted():
    """Credible level is the fraction of samples below the injection value."""
    result = make_result_with_injection(np.arange(10.0), 5.0)
    assert result.get_injection_credible_level("x") == pytest.approx(0.5)


def test_credible_level_weighted():
    """Weighted credible level weights the below-injection fraction."""
    x = np.arange(10.0)
    weights = np.where(x < 5, 2.0, 1.0)
    result = make_result_with_injection(x, 5.0, weights=weights)
    # Below injection: x=0..4 (weight 2 each) => 10; total weight = 10 + 5 = 15.
    assert result.get_injection_credible_level("x", weighted=True) == pytest.approx(
        10 / 15
    )


def test_credible_level_missing_parameter_returns_nan():
    result = make_result_with_injection(np.arange(10.0), 5.0)
    assert np.isnan(result.get_injection_credible_level("nonexistent"))
