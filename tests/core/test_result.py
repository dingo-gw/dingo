import math

import numpy as np
import pandas as pd
import pytest
from bilby.core.prior import Constraint, DeltaFunction, PriorDict, Uniform
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
        assert (
            counts[i] <= counts[i + 1]
        ), f"k={i+1} gave {counts[i]} samples but k={i+2} gave {counts[i+1]}"


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
# Statistical / metadata properties
# ---------------------------------------------------------------------------


def make_result(samples=None, settings=None, event_metadata=None):
    """Construct a minimal core Result, optionally with settings/event_metadata."""
    dictionary = {}
    if samples is not None:
        dictionary["samples"] = samples
    if settings is not None:
        dictionary["settings"] = settings
    result = Result(dictionary=dictionary)
    if event_metadata is not None:
        result.event_metadata = event_metadata
    return result


def test_num_samples():
    result = make_result(pd.DataFrame({"x": np.arange(7.0)}))
    assert result.num_samples == 7


def test_num_samples_zero_when_no_samples():
    result = make_result()
    assert result.num_samples == 0


def test_effective_sample_size_uniform_weights_equals_n():
    result = make_result_with_weights(np.ones(10))
    assert result.effective_sample_size == pytest.approx(10.0)
    assert result.n_eff == result.effective_sample_size


def test_effective_sample_size_matches_formula():
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    result = make_result_with_weights(weights)
    expected = np.sum(weights) ** 2 / np.sum(weights**2)
    assert result.effective_sample_size == pytest.approx(expected)


def test_effective_sample_size_none_without_weights():
    result = make_result(pd.DataFrame({"x": [1.0, 2.0]}))
    assert result.effective_sample_size is None


def test_sample_efficiency():
    result = make_result_with_weights(np.ones(8))
    assert result.sample_efficiency == pytest.approx(1.0)  # uniform -> n_eff / N = 1


def test_sample_efficiency_none_without_weights():
    result = make_result(pd.DataFrame({"x": [1.0, 2.0]}))
    assert result.sample_efficiency is None


def test_log_evidence_std_requires_weights_and_log_evidence():
    result = make_result_with_weights([1.0, 2.0, 3.0])
    # No log_evidence set yet.
    assert result.log_evidence_std is None
    result.log_evidence = -5.0
    assert result.log_evidence_std is not None
    assert result.log_evidence_std > 0


def test_log_bayes_factor():
    result = make_result(pd.DataFrame({"x": [1.0]}))
    assert result.log_bayes_factor is None
    result.log_evidence = -3.0
    result.log_noise_evidence = -10.0
    assert result.log_bayes_factor == pytest.approx(7.0)


def test_injection_parameters():
    result = make_result(pd.DataFrame({"x": [1.0]}))
    assert result.injection_parameters is None
    result.event_metadata = {"injection_parameters": {"mass": 30.0}}
    assert result.injection_parameters == {"mass": 30.0}


def test_metadata_and_base_metadata():
    settings = {"train_settings": {"data": {}}, "value": 1}
    result = make_result(pd.DataFrame({"x": [1.0]}), settings=settings)
    assert result.metadata is result.settings
    # Non-unconditional model: base_metadata is the full metadata.
    assert result.base_metadata is result.metadata


def test_base_metadata_unconditional_returns_base():
    base = {"some": "precursor"}
    settings = {"train_settings": {"data": {"unconditional": True}}, "base": base}
    result = make_result(pd.DataFrame({"x": [1.0]}), settings=settings)
    assert result.base_metadata is base


def test_parameter_subset_keeps_only_requested_columns():
    samples = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "log_prob": [0.0, 0.0]})
    result = make_result(samples)
    subset = result.parameter_subset(["a"])
    assert list(subset.samples.columns) == ["a"]
    assert isinstance(subset, Result)


# ---------------------------------------------------------------------------
# _calculate_evidence
# ---------------------------------------------------------------------------


def test_calculate_evidence_matches_logsumexp():
    n = 6
    samples = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "log_prob": np.linspace(0.0, 1.0, n),
            "log_prior": np.full(n, -2.0),
            "log_likelihood": np.linspace(-1.0, 1.0, n),
        }
    )
    result = make_result(samples)
    result._calculate_evidence()

    log_weights = samples["log_prior"] + samples["log_likelihood"] - samples["log_prob"]
    expected = logsumexp(log_weights) - np.log(n)
    assert result.log_evidence == pytest.approx(expected)
    # Stored weights are normalized to mean 1.
    assert result.samples["weights"].mean() == pytest.approx(1.0)


def test_calculate_evidence_includes_delta_log_prob_target():
    n = 4
    delta = np.array([0.0, 0.5, -0.5, 1.0])
    samples = pd.DataFrame(
        {
            "log_prob": np.zeros(n),
            "log_prior": np.zeros(n),
            "log_likelihood": np.zeros(n),
            "delta_log_prob_target": delta,
        }
    )
    result = make_result(samples)
    result._calculate_evidence()
    expected = logsumexp(delta) - np.log(n)
    assert result.log_evidence == pytest.approx(expected)


# ---------------------------------------------------------------------------
# importance_sample orchestration (stubbed prior + likelihood)
# ---------------------------------------------------------------------------


class _StubResult(Result):
    """Result with an analytic prior and a trivial likelihood, to exercise the
    importance_sample orchestration without any GW machinery."""

    def _build_prior(self):
        self.prior = PriorDict(
            {
                "a": Uniform(0.0, 1.0, name="a"),
                "b": Uniform(0.0, 1.0, name="b"),
                "c": DeltaFunction(0.5, name="c"),  # fixed parameter
            }
        )

    def _build_likelihood(self, **likelihood_kwargs):
        class _StubLikelihood:
            log_Zn = -10.0

            def log_likelihood_multi(self, theta, num_processes=1):
                return np.zeros(len(theta))

        self.likelihood = _StubLikelihood()


def _stub_samples(n=5, with_log_prob=True):
    data = {
        "a": np.linspace(0.1, 0.9, n),
        "b": np.linspace(0.1, 0.9, n),
        "c": np.full(n, 0.5),
    }
    if with_log_prob:
        data["log_prob"] = np.zeros(n)
    return pd.DataFrame(data)


def test_importance_sample_populates_columns_and_evidence():
    result = _StubResult(dictionary={"samples": _stub_samples()})
    result.importance_sample(num_processes=1)
    for col in ("log_prior", "log_likelihood", "weights"):
        assert col in result.samples.columns
    assert result.log_evidence is not None
    assert result.log_noise_evidence == -10.0


def test_importance_sample_requires_samples():
    result = _StubResult(dictionary={"samples": _stub_samples()})
    result.samples = None
    with pytest.raises(KeyError):
        result.importance_sample()


def test_importance_sample_requires_log_prob():
    result = _StubResult(dictionary={"samples": _stub_samples(with_log_prob=False)})
    with pytest.raises(KeyError, match="log probability"):
        result.importance_sample()


# ---------------------------------------------------------------------------
# split / merge
# ---------------------------------------------------------------------------


def _result_with_evidence_columns(n=10):
    samples = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "log_prob": np.zeros(n),
            "log_prior": np.zeros(n),
            "log_likelihood": np.zeros(n),
        }
    )
    return Result(
        dictionary={"samples": samples, "settings": {"train_settings": {"data": {}}}}
    )


def test_split_partitions_samples():
    result = _result_with_evidence_columns(10)
    parts = result.split(3)
    assert len(parts) == 3
    assert sum(p.num_samples for p in parts) == 10
    assert all(isinstance(p, Result) for p in parts)


def test_split_then_merge_round_trips():
    result = _result_with_evidence_columns(10)
    merged = Result.merge(result.split(3))
    np.testing.assert_array_equal(
        merged.samples["x"].to_numpy(), result.samples["x"].to_numpy()
    )


def test_merge_incompatible_metadata_raises():
    result = _result_with_evidence_columns(10)
    parts = result.split(2)
    parts[1].settings = {"train_settings": {"data": {"changed": True}}}
    with pytest.raises(ValueError, match="same metadata"):
        Result.merge(parts)


# ---------------------------------------------------------------------------
# sampling_importance_resampling
# ---------------------------------------------------------------------------


def test_sampling_importance_resampling_count_and_drops_weights():
    result = make_result_with_weights([1.0, 2.0, 3.0, 0.5, 4.0], extra_col=False)
    out = result.sampling_importance_resampling(num_samples=3, random_state=0)
    assert len(out) == 3
    assert "weights" not in out.columns


def test_sampling_importance_resampling_too_many_raises():
    result = make_result_with_weights([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Cannot sample more"):
        result.sampling_importance_resampling(num_samples=100)


# ---------------------------------------------------------------------------
# prior-derived parameter-key properties
# ---------------------------------------------------------------------------


def test_parameter_key_properties_partition_the_prior():
    result = Result(dictionary={"samples": pd.DataFrame({"a": [0.5]})})
    # core Result._build_prior leaves prior=None; set a prior with one of each kind.
    result.prior = PriorDict(
        {
            "a": Uniform(0.0, 1.0, name="a"),  # search parameter
            "b": Constraint(0.0, 1.0, name="b"),  # constraint
            "c": DeltaFunction(0.5, name="c"),  # fixed parameter
        }
    )
    assert result.search_parameter_keys == ["a"]
    assert result.constraint_parameter_keys == ["b"]
    assert result.fixed_parameter_keys == ["c"]


# ---------------------------------------------------------------------------
# print_summary (smoke)
# ---------------------------------------------------------------------------


def test_print_summary_runs_with_and_without_evidence(capsys):
    result = make_result_with_weights([1.0, 2.0, 3.0])
    result.print_summary()  # no log_evidence yet
    result.log_evidence = -3.0
    result.print_summary()  # with evidence / n_eff / efficiency
    assert "Number of samples" in capsys.readouterr().out
