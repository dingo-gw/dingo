"""Regression test for the DeltaFunction within-prior filtering fix in result.py.

Guards against the bug where computing log_prior over ALL prior params (including
DeltaFunction params) would drive log_prior to -inf for all samples, causing the
within_prior mask to exclude every sample.

Background: in sample_synthetic_phase, RA corrections (trigger_time vs model
ref_time) can shift fixed/DeltaFunction parameters by tiny amounts.  When
PriorDict.ln_prob is called over a DeltaFunction param with a value that is
even slightly off-peak, it returns -inf, zeroing out the entire within-prior
mask.  The fix is to compute log_prior only over non-DeltaFunction,
non-Constraint params.

This test exercises the same bilby PriorDict operations used in
dingo.gw.result.Result.sample_synthetic_phase without requiring any model,
waveform, or GPU fixtures.
"""

import numpy as np
import pandas as pd
from bilby.core.prior import Constraint, DeltaFunction, PriorDict, Uniform


def _log_prior_all_params(prior, samples):
    """Old (buggy) approach: compute ln_prob over ALL non-Constraint params,
    including DeltaFunction params."""
    param_keys = [k for k, v in prior.items() if not isinstance(v, Constraint)]
    return prior.ln_prob(samples[param_keys], axis=0)


def _log_prior_continuous_only(prior, samples):
    """Fixed approach: compute ln_prob over non-Constraint, non-DeltaFunction params
    only (mirrors sample_synthetic_phase in dingo/gw/result.py)."""
    prior_keys_for_lp = [
        k
        for k, v in prior.items()
        if not isinstance(v, Constraint) and not isinstance(v, DeltaFunction)
    ]
    return prior.ln_prob(samples[prior_keys_for_lp], axis=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fixed_approach_retains_samples_when_delta_param_is_shifted():
    """Core regression: with a DeltaFunction param shifted slightly off-peak
    (as happens after RA trigger-time correction), the fixed approach must NOT
    produce -inf log_prior, while the old approach does."""
    prior = PriorDict(
        {
            "mass": Uniform(minimum=10.0, maximum=50.0, name="mass"),
            "phase": DeltaFunction(peak=0.0, name="phase"),
        }
    )

    rng = np.random.default_rng(42)
    n = 50
    samples = pd.DataFrame(
        {
            "mass": rng.uniform(10.0, 50.0, n),
            # Tiny shift from peak=0.0, simulating RA trigger-time correction.
            "phase": np.full(n, 1e-8),
        }
    )

    lp_fixed = _log_prior_continuous_only(prior, samples)
    within_prior_fixed = np.isfinite(lp_fixed)

    lp_buggy = _log_prior_all_params(prior, samples)
    within_prior_buggy = np.isfinite(lp_buggy)

    # Fixed approach: all samples have mass in [10,50], so all should have finite
    # log_prior.  DeltaFunction param is excluded from computation.
    assert within_prior_fixed.all(), (
        f"Fixed approach: {(~within_prior_fixed).sum()}/{n} samples have non-finite "
        "log_prior. DeltaFunction params must not be included in log_prior computation."
    )

    # Buggy approach: DeltaFunction.ln_prob(1e-8) == -inf for all samples.
    # This documents the regression scenario — the old code would have dropped
    # every sample from within_prior.
    assert not within_prior_buggy.any(), (
        "Expected old approach to produce -inf log_prior when DeltaFunction param "
        "is off-peak, but got finite values. The regression scenario may have changed."
    )


def test_fixed_approach_still_excludes_out_of_bounds_continuous_samples():
    """Samples outside the continuous Uniform prior must still be excluded,
    even though DeltaFunction params are skipped in the log_prior computation."""
    prior = PriorDict(
        {
            "mass": Uniform(minimum=10.0, maximum=50.0, name="mass"),
            "phase": DeltaFunction(peak=0.0, name="phase"),
        }
    )

    samples = pd.DataFrame(
        {
            #         in-bounds, out-of-bounds, in-bounds, out-of-bounds
            "mass": [20.0, 100.0, 15.0, 5.0],
            "phase": [0.0, 0.0, 0.0, 0.0],
        }
    )

    lp = _log_prior_continuous_only(prior, samples)
    within_prior = np.isfinite(lp)
    expected = np.array([True, False, True, False])

    np.testing.assert_array_equal(
        within_prior,
        expected,
        err_msg="Fixed within-prior mask doesn't correctly exclude OOB continuous samples.",
    )
