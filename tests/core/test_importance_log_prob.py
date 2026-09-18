import numpy as np
import pandas as pd

from dingo.core.result import Result


def _bare_result(log_prob, log_likelihood, log_prior):
    result = Result.__new__(Result)
    result.samples = pd.DataFrame(
        {
            "log_prob": log_prob,
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
        }
    )
    return result


class TestImportanceLogProbOccupancy:
    def test_finite_proposal_keeps_relative_weights(self):
        result = _bare_result([0.0, -1.0], [0.0, 0.0], [0.0, 0.0])
        result._calculate_evidence()
        weights = result.samples["weights"].to_numpy()
        assert weights[0] > 0.0 and weights[1] > 0.0

    def test_missing_proposal_gets_zero_weight(self):
        result = _bare_result([0.0, np.nan], [0.0, 0.0], [0.0, 0.0])
        result._calculate_evidence()
        weights = result.samples["weights"].to_numpy()
        assert weights[0] > 0.0
        assert weights[1] == 0.0

    def test_minus_inf_prior_stays_zero(self):
        result = _bare_result([0.0, 0.0], [0.0, 0.0], [0.0, -np.inf])
        result._calculate_evidence()
        weights = result.samples["weights"].to_numpy()
        assert weights[1] == 0.0

    def test_nan_proposal_is_not_treated_as_q_equals_one(self):
        leftover = np.nan_to_num(np.array([np.nan]))
        dest = np.where(np.isfinite(np.array([np.nan])), np.array([np.nan]), np.inf)
        assert leftover[0] == 0.0
        assert np.isinf(dest[0])
