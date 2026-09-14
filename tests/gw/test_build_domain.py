import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.domains.build_domain import (
    build_domain,
    build_domain_from_model_metadata,
)


def test_build_domain_uniform_frequency():
    settings = {
        "type": "UniformFrequencyDomain",
        "f_min": 20.0,
        "f_max": 1024.0,
        "delta_f": 0.25,
    }
    domain = build_domain(settings)
    assert isinstance(domain, UniformFrequencyDomain)
    assert domain.f_min == 20.0 and domain.f_max == 1024.0


def test_build_domain_frequency_domain_alias():
    # "FD" and "FrequencyDomain" are aliases for UniformFrequencyDomain.
    domain = build_domain({"type": "FD", "f_min": 20.0, "f_max": 512.0, "delta_f": 0.5})
    assert isinstance(domain, UniformFrequencyDomain)


def test_build_domain_missing_type_raises():
    with pytest.raises(ValueError, match='"type"'):
        build_domain({"f_min": 20.0, "f_max": 1024.0, "delta_f": 0.25})


def test_build_domain_unknown_type_raises():
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_domain({"type": "NotADomain"})


# ---------------------------------------------------------------------------
# build_domain_from_model_metadata
# ---------------------------------------------------------------------------

_METADATA = {
    "dataset_settings": {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.25,
        }
    },
    "train_settings": {"data": {}},
}


def test_build_domain_from_model_metadata():
    domain = build_domain_from_model_metadata(_METADATA)
    assert isinstance(domain, UniformFrequencyDomain)
    assert domain.f_min == 20.0


def test_build_domain_from_model_metadata_applies_domain_update():
    metadata = {
        **_METADATA,
        "train_settings": {"data": {"domain_update": {"f_min": 30.0}}},
    }
    assert build_domain_from_model_metadata(metadata).f_min == 30.0


def test_build_domain_from_model_metadata_base_is_noop_for_uniform_domain():
    # A UniformFrequencyDomain has no base_domain, so base=True returns it unchanged.
    domain = build_domain_from_model_metadata(_METADATA, base=True)
    assert isinstance(domain, UniformFrequencyDomain)
