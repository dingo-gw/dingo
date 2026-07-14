import numpy as np
import pytest
from astropy.time import Time
from astropy.utils import iers

from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.inference.gw_samplers import (
    GWSampler,
    check_frequency_updates,
    _validate_maximum_frequency,
    _validate_minimum_frequency,
)

# Avoid network access (and the associated timeout) when astropy computes sidereal
# time in _correct_reference_time; the bundled IERS data is sufficient for tests.
iers.conf.auto_download = False


DETECTORS = ["H1", "L1"]
INFERENCE_PARAMETERS = ["chirp_mass", "mass_ratio", "ra", "dec"]

DOMAIN_SETTINGS = {
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 1024.0,
    "delta_f": 0.25,
}


# ---------------------------------------------------------------------------
# Frequency-range validators (pure functions, only need a domain).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GWSamplerMixin methods (lightweight GWSampler; network not exercised).
# ---------------------------------------------------------------------------


def _build_gw_sampler(unconditional=False, domain_update=None):
    """Build a GWSampler around a tiny flow plus minimal but valid GW metadata.

    The network is never run by the methods under test here; it only needs to exist.
    """
    standardization = {
        "mean": {p: 0.0 for p in INFERENCE_PARAMETERS},
        "std": {p: 1.0 for p in INFERENCE_PARAMETERS},
    }
    posterior_kwargs = {
        "input_dim": len(INFERENCE_PARAMETERS),
        "context_dim": None,
        "num_flow_steps": 2,
        "base_transform_kwargs": {
            "hidden_dim": 8,
            "num_transform_blocks": 1,
            "activation": "elu",
            "dropout_probability": 0.0,
            "batch_norm": False,
            "num_bins": 4,
            "base_transform_type": "rq-coupling",
        },
    }
    data_settings = {
        "unconditional": unconditional,
        "inference_parameters": INFERENCE_PARAMETERS,
        "standardization": standardization,
        "detectors": DETECTORS,
        "ref_time": 1126259462.4,
        "extrinsic_prior": {
            "dec": "default",
            "ra": "default",
            "geocent_time": "default",
            "luminosity_distance": "default",
            "psi": "default",
        },
    }
    if domain_update is not None:
        data_settings["domain_update"] = domain_update

    metadata = {
        "train_settings": {
            "model": {
                "posterior_model_type": "normalizing_flow",
                "posterior_kwargs": posterior_kwargs,
            },
            "data": data_settings,
        },
        "dataset_settings": {
            "domain": DOMAIN_SETTINGS,
            "intrinsic_prior": {
                "mass_1": "bilby.core.prior.Constraint(minimum=10, maximum=80)",
                "mass_2": "bilby.core.prior.Constraint(minimum=10, maximum=80)",
                "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass("
                "minimum=25, maximum=31)",
                "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio("
                "minimum=0.125, maximum=1)",
                "phase": "default",
                "a_1": 0.0,
                "a_2": 0.0,
            },
        },
    }
    if unconditional:
        metadata["base"] = metadata
    model = NormalizingFlowPosteriorModel(metadata=metadata, device="cpu")
    return GWSampler(model=model)


@pytest.fixture()
def gw_sampler():
    return _build_gw_sampler()


def test_build_domain_from_metadata(gw_sampler):
    assert isinstance(gw_sampler.domain, UniformFrequencyDomain)
    assert gw_sampler.domain.f_min == DOMAIN_SETTINGS["f_min"]
    assert gw_sampler.domain.f_max == DOMAIN_SETTINGS["f_max"]


def test_build_domain_applies_domain_update():
    sampler = _build_gw_sampler(domain_update={"f_min": 30.0})
    assert sampler.domain.f_min == 30.0


def test_correct_reference_time_round_trip(gw_sampler):
    gw_sampler._event_metadata = {"time_event": gw_sampler.t_ref + 3600.0}
    samples = {"ra": np.array([0.5, 1.5, 2.5]), "dec": np.array([0.1, 0.2, 0.3])}
    original_ra = samples["ra"].copy()

    gw_sampler._correct_reference_time(samples, inverse=False)
    assert not np.allclose(samples["ra"], original_ra)
    assert np.all((samples["ra"] >= 0) & (samples["ra"] < 2 * np.pi))

    gw_sampler._correct_reference_time(samples, inverse=True)
    np.testing.assert_allclose(samples["ra"], original_ra)


def test_correct_reference_time_matches_sidereal_shift(gw_sampler):
    """The RA shift must equal the difference in apparent sidereal time."""
    t_event = gw_sampler.t_ref + 3600.0
    gw_sampler._event_metadata = {"time_event": t_event}
    samples = {"ra": np.array([0.5, 1.5, 2.5])}
    original_ra = samples["ra"].copy()

    ra_correction = (
        Time(t_event, format="gps", scale="utc").sidereal_time("apparent", "greenwich")
        - Time(gw_sampler.t_ref, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
    ).rad

    gw_sampler._correct_reference_time(samples, inverse=False)
    np.testing.assert_allclose(
        samples["ra"], (original_ra + ra_correction) % (2 * np.pi)
    )


def test_correct_reference_time_noop_when_time_matches_reference(gw_sampler):
    gw_sampler._event_metadata = {"time_event": gw_sampler.t_ref}
    samples = {"ra": np.array([0.5, 1.5])}
    original_ra = samples["ra"].copy()
    gw_sampler._correct_reference_time(samples, inverse=False)
    np.testing.assert_allclose(samples["ra"], original_ra)


def test_correct_reference_time_noop_without_ra(gw_sampler):
    gw_sampler._event_metadata = {"time_event": gw_sampler.t_ref + 3600.0}
    samples = {"dec": np.array([0.1, 0.2])}
    # No "ra" key: nothing to correct, and no error.
    gw_sampler._correct_reference_time(samples, inverse=False)
    assert "ra" not in samples


def test_post_process_forward_adds_fixed_prior_parameters(gw_sampler):
    samples = {p: np.zeros(5) for p in INFERENCE_PARAMETERS}
    gw_sampler._event_metadata = None
    gw_sampler._post_process(samples, inverse=False)
    # a_1 and a_2 are DeltaFunctions (0.0) in the intrinsic prior; they get added.
    for fixed in ("a_1", "a_2"):
        assert fixed in samples
        np.testing.assert_array_equal(samples[fixed], np.zeros(5))


def test_post_process_inverse_drops_non_inference_parameters(gw_sampler):
    samples = {
        "chirp_mass": np.array([28.0]),
        "ra": np.array([1.0]),
        "log_prob": np.array([0.5]),
        "extra": np.array([9.0]),
    }
    gw_sampler._event_metadata = None
    gw_sampler._post_process(samples, inverse=True)
    assert set(samples) <= set(INFERENCE_PARAMETERS)
    assert "log_prob" not in samples
    assert "extra" not in samples


def test_frequency_updates_flag(gw_sampler):
    # By default the requested range equals the domain, so no updates are flagged and
    # the min/max frequencies report the domain bounds.
    assert gw_sampler.minimum_frequency == gw_sampler.domain.f_min
    assert gw_sampler.maximum_frequency == gw_sampler.domain.f_max
    assert gw_sampler.frequency_updates is False

    # A requested minimum frequency that differs from the domain flags an update.
    # (Set the private attribute directly to bypass the validating setter, which
    # would also rebuild the transforms.)
    gw_sampler._minimum_frequency = 40.0
    assert gw_sampler.frequency_updates is True


def test_event_metadata_injects_frequency_bounds(gw_sampler):
    metadata = gw_sampler.event_metadata
    assert metadata["minimum_frequency"] == gw_sampler.domain.f_min
    assert metadata["maximum_frequency"] == gw_sampler.domain.f_max
