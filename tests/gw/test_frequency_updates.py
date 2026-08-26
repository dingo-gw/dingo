"""Unit tests for the frequency-range validators (``dingo.gw.frequency_updates``):
the pure functions that gate event- and importance-sampling-time minimum/maximum
frequency updates against the network's training domain and its random-strain
cropping licence.

These are called at INI-parse time (``dingo_pipe``) and, via ``GWSamplerContext``,
whenever event metadata carries ``minimum_frequency`` / ``maximum_frequency``. The
context exercises them indirectly through ``prepared_data`` (see
``test_gw_sampler_context``); this file covers the functions directly, including the
per-detector cap/floor, detector-key, independent-detector, and cropping-licence
branches the context-level tests do not reach.

(Adapted from the validator block of the pre-sampler-revamp ``test_gw_samplers.py``,
which imported them from the now-deleted ``dingo.gw.inference.gw_samplers``.)
"""

import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.frequency_updates import (
    check_frequency_updates,
    _validate_maximum_frequency,
    _validate_minimum_frequency,
)


DETECTORS = ["H1", "L1"]

DOMAIN_SETTINGS = {
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 1024.0,
    "delta_f": 0.25,
}


@pytest.fixture()
def domain():
    return UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.25)


@pytest.mark.parametrize(
    "validate, valid_change, beyond_bound",
    [
        (_validate_minimum_frequency, 40.0, 10.0),  # raise f_min; below hard f_min
        (_validate_maximum_frequency, 512.0, 2048.0),  # lower f_max; above hard f_max
    ],
)
def test_frequency_validator_no_op_when_unchanged(
    domain, validate, valid_change, beyond_bound
):
    # Value equal to the domain bound is a no-op and is allowed even without cropping.
    bound = domain.f_min if validate is _validate_minimum_frequency else domain.f_max
    assert validate(bound, DETECTORS, domain, None) is None


@pytest.mark.parametrize(
    "validate, valid_change, beyond_bound",
    [
        (_validate_minimum_frequency, 40.0, 10.0),
        (_validate_maximum_frequency, 512.0, 2048.0),
    ],
)
def test_frequency_validator_expands_float_to_all_detectors(
    domain, validate, valid_change, beyond_bound
):
    # A float applies to every detector; a valid change passes with cropping on.
    # The cap/floor must be given explicitly, else it defaults to the domain bound.
    crop = {"cropping_probability": 0.5, "f_min_upper": 100.0, "f_max_lower": 400.0}
    assert validate(valid_change, DETECTORS, domain, crop) is None


@pytest.mark.parametrize(
    "validate, valid_change, beyond_bound",
    [
        (_validate_minimum_frequency, 40.0, 10.0),
        (_validate_maximum_frequency, 512.0, 2048.0),
    ],
)
def test_frequency_validator_rejects_value_beyond_hard_bound(
    domain, validate, valid_change, beyond_bound
):
    crop = {"cropping_probability": 0.5}
    with pytest.raises(ValueError, match="domain.f_"):
        validate(beyond_bound, DETECTORS, domain, crop)


@pytest.mark.parametrize(
    "validate, valid_change",
    [(_validate_minimum_frequency, 40.0), (_validate_maximum_frequency, 512.0)],
)
def test_frequency_validator_rejects_detector_key_mismatch(
    domain, validate, valid_change
):
    crop = {"cropping_probability": 0.5}
    with pytest.raises(ValueError, match="exactly detectors"):
        validate({"H1": valid_change}, DETECTORS, domain, crop)


@pytest.mark.parametrize(
    "validate, valid_change",
    [(_validate_minimum_frequency, 40.0), (_validate_maximum_frequency, 512.0)],
)
def test_frequency_validator_rejects_change_when_cropping_disabled(
    domain, validate, valid_change
):
    # No crop settings at all.
    with pytest.raises(ValueError, match="[Cc]ropping"):
        validate(valid_change, DETECTORS, domain, None)
    # Crop settings present but probability zero.
    with pytest.raises(ValueError, match="[Cc]ropping"):
        validate(valid_change, DETECTORS, domain, {"cropping_probability": 0.0})


def test_validate_minimum_frequency_rejects_value_above_cap(domain):
    crop = {"cropping_probability": 0.5, "f_min_upper": 60.0}
    assert _validate_minimum_frequency(50.0, DETECTORS, domain, crop) is None
    with pytest.raises(ValueError, match="upper bound"):
        _validate_minimum_frequency(80.0, DETECTORS, domain, crop)


def test_validate_maximum_frequency_rejects_value_below_floor(domain):
    crop = {"cropping_probability": 0.5, "f_max_lower": 400.0}
    assert _validate_maximum_frequency(500.0, DETECTORS, domain, crop) is None
    with pytest.raises(ValueError, match="lower bound"):
        _validate_maximum_frequency(300.0, DETECTORS, domain, crop)


def test_validate_minimum_frequency_rejects_differing_values_when_not_independent(
    domain,
):
    crop = {
        "cropping_probability": 0.5,
        "independent_detectors": False,
        "f_min_upper": 100.0,
    }
    with pytest.raises(ValueError, match="[Ii]ndependent"):
        _validate_minimum_frequency({"H1": 40.0, "L1": 50.0}, DETECTORS, domain, crop)


def test_check_frequency_updates_accepts_valid_and_rejects_invalid():
    model_metadata = {
        "train_settings": {
            "data": {
                "detectors": DETECTORS,
                "random_strain_cropping": {
                    "cropping_probability": 0.5,
                    "f_min_upper": 100.0,
                    "f_max_lower": 400.0,
                },
            }
        },
        "dataset_settings": {"domain": DOMAIN_SETTINGS},
    }
    # Valid frequency updates pass without raising.
    assert check_frequency_updates(model_metadata, f_min=40.0, f_max=512.0) is None
    # Beyond the hard bound raises.
    with pytest.raises(ValueError, match="domain.f_min"):
        check_frequency_updates(model_metadata, f_min=10.0)
