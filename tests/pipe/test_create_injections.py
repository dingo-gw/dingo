import bilby
import pytest

from dingo.pipe.create_injections import get_time_prior


def test_get_time_prior_zero_uncertainty_is_delta_function():
    prior = get_time_prior(1126259462.4, 0.0)
    assert isinstance(prior, bilby.core.prior.DeltaFunction)
    assert prior.peak == 1126259462.4


def test_get_time_prior_positive_uncertainty_spans_time():
    prior = get_time_prior(1000.0, 0.1)
    assert not isinstance(prior, bilby.core.prior.DeltaFunction)
    assert prior.minimum <= 1000.0 <= prior.maximum


def test_get_time_prior_negative_uncertainty_raises():
    with pytest.raises(ValueError, match="uncertainty"):
        get_time_prior(1000.0, -0.1)
