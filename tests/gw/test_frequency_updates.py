"""Unit tests for the per-event-setting validators (``dingo.gw.frequency_updates``):
the pure functions that gate event- and importance-sampling-time minimum/maximum
frequency updates, PSD notches, and detector subsets against the network's training
domain and its training-time licenses (random strain cropping, or the token masks of
a tokenized network).

These are called at INI-parse time (``dingo_pipe``) and, via ``GWSamplerContext``,
whenever event metadata carries the corresponding keys. The context exercises them
indirectly through ``prepared_data`` (see ``test_gw_sampler_context``); this file
covers the functions directly.
"""

import warnings

import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.frequency_updates import (
    check_detector_update,
    check_frequency_updates,
    check_importance_sampling_frequency_range,
    check_psd_notches,
    resolve_frequency_bounds,
    _validate_detectors_transformer,
    _validate_frequency_bound,
    _validate_psd_notches,
)


DETECTORS = ["H1", "L1"]

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


def _crop_data_settings(**crop):
    return {"detectors": DETECTORS, "random_strain_cropping": crop}


@pytest.mark.parametrize("bound", ["minimum_frequency", "maximum_frequency"])
def test_frequency_bound_no_op_when_unchanged(domain, bound):
    """Domain values always pass, even without any frequency flexibility."""
    value = domain.f_min if bound == "minimum_frequency" else domain.f_max
    _validate_frequency_bound(value, bound, domain, {"detectors": DETECTORS})


@pytest.mark.parametrize(
    "bound, invalid",
    [("minimum_frequency", 10.0), ("maximum_frequency", 2048.0)],
)
def test_frequency_bound_rejects_value_beyond_hard_bound(domain, bound, invalid):
    settings = _crop_data_settings(cropping_probability=0.8)
    with pytest.raises(ValueError, match="domain.f_"):
        _validate_frequency_bound(invalid, bound, domain, settings)


@pytest.mark.parametrize(
    "bound, changed",
    [("minimum_frequency", 40.0), ("maximum_frequency", 512.0)],
)
def test_frequency_bound_rejects_change_without_flexibility(domain, bound, changed):
    with pytest.raises(ValueError, match="not trained with variable"):
        _validate_frequency_bound(changed, bound, domain, {"detectors": DETECTORS})


@pytest.mark.parametrize(
    "bound, changed",
    [("minimum_frequency", 40.0), ("maximum_frequency", 512.0)],
)
def test_frequency_bound_rejects_change_when_cropping_disabled(domain, bound, changed):
    settings = _crop_data_settings(cropping_probability=0.0)
    with pytest.raises(ValueError, match="[Cc]ropping"):
        _validate_frequency_bound(changed, bound, domain, settings)


def test_frequency_bound_unknown_detector_key_raises(domain):
    settings = _crop_data_settings(cropping_probability=0.8, f_min_upper=64.0)
    with pytest.raises(ValueError, match="not.*trained with"):
        _validate_frequency_bound({"K1": 40.0}, "minimum_frequency", domain, settings)


def test_frequency_bound_rejects_value_above_cropping_cap(domain):
    settings = _crop_data_settings(cropping_probability=0.8, f_min_upper=64.0)
    _validate_frequency_bound(50.0, "minimum_frequency", domain, settings)
    with pytest.raises(ValueError, match="f_min_upper"):
        _validate_frequency_bound(80.0, "minimum_frequency", domain, settings)


def test_frequency_bound_rejects_value_below_cropping_floor(domain):
    settings = _crop_data_settings(cropping_probability=0.8, f_max_lower=400.0)
    _validate_frequency_bound(500.0, "maximum_frequency", domain, settings)
    with pytest.raises(ValueError, match="f_max_lower"):
        _validate_frequency_bound(300.0, "maximum_frequency", domain, settings)


def test_frequency_bound_absent_cropping_cap_rejects_any_change(domain):
    """No f_min_upper in the settings means the lower side was never cropped."""
    settings = _crop_data_settings(cropping_probability=0.8, f_max_lower=400.0)
    with pytest.raises(ValueError, match="f_min_upper"):
        _validate_frequency_bound(40.0, "minimum_frequency", domain, settings)


def test_frequency_bound_rejects_differing_values_when_not_independent(domain):
    settings = _crop_data_settings(
        cropping_probability=0.8, f_min_upper=64.0, independent_detectors=False
    )
    with pytest.raises(ValueError, match="[Ii]ndependent"):
        _validate_frequency_bound(
            {"H1": 40.0, "L1": 50.0}, "minimum_frequency", domain, settings
        )


def test_frequency_bound_partial_dict_not_independent_raises(domain):
    """A partial dict changing one detector implies unequal bounds."""
    settings = _crop_data_settings(
        cropping_probability=0.8, f_min_upper=64.0, independent_detectors=False
    )
    with pytest.raises(ValueError, match="[Ii]ndependent"):
        _validate_frequency_bound({"H1": 40.0}, "minimum_frequency", domain, settings)


def test_frequency_bound_range_absent_key_rejects_change(domain):
    """mask_frequency_range without f_min_upper means lower cuts were never drawn."""
    settings = {
        "detectors": DETECTORS,
        "tokenization": {"mask_frequency_range": {"f_max_lower": 80.0}},
    }
    with pytest.raises(ValueError, match="f_min_upper"):
        _validate_frequency_bound(40.0, "minimum_frequency", domain, settings)


def test_frequency_bound_random_tokens_only_warns(domain):
    settings = {
        "detectors": DETECTORS,
        "tokenization": {"mask_random_tokens": {"p_mask": 0.4, "max_num_tokens": 10}},
    }
    with pytest.warns(UserWarning, match="mask_random_tokens"):
        _validate_frequency_bound(40.0, "minimum_frequency", domain, settings)


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
# check_detector_update / _validate_detectors_transformer
# ---------------------------------------------------------------------------


def _make_metadata(detectors, mask_detectors=None, mask_random_tokens=None):
    tok = {}
    if mask_detectors is not None:
        tok["mask_detectors"] = mask_detectors
    if mask_random_tokens is not None:
        tok["mask_random_tokens"] = mask_random_tokens
    data = {"detectors": detectors}
    if tok:
        data["tokenization"] = tok
    return {"train_settings": {"data": data}}


# --- _validate_detectors_transformer ---

HLV_SETTINGS = {
    "p_num_masked": [0.6, 0.3, 0.1],
    "p_detector": {"H1": 0.3, "L1": 0.3, "V1": 0.4},
}

HL_SETTINGS = {
    "p_num_masked": [0.6, 0.4],
    "p_detector": {"H1": 0.5, "L1": 0.5},
}


def test_validate_full_detector_set_allowed():
    _validate_detectors_transformer(
        ["H1", "L1", "V1"], ["H1", "L1", "V1"], HLV_SETTINGS
    )


def test_validate_hl_subset_of_hlv_allowed():
    _validate_detectors_transformer(["H1", "L1"], ["H1", "L1", "V1"], HLV_SETTINGS)


def test_validate_single_detector_subset_allowed():
    _validate_detectors_transformer(["H1"], ["H1", "L1", "V1"], HLV_SETTINGS)


def test_validate_two_detector_model_full_set():
    _validate_detectors_transformer(["H1", "L1"], ["H1", "L1"], HL_SETTINGS)


def test_validate_two_detector_model_single_detector():
    _validate_detectors_transformer(["H1"], ["H1", "L1"], HL_SETTINGS)


def test_validate_event_not_subset_raises():
    with pytest.raises(ValueError, match="only trained with"):
        _validate_detectors_transformer(["H1", "V1"], ["H1", "L1"], HL_SETTINGS)


def test_validate_p_mask_zero_for_count_raises():
    # p_num_masked[0] = 0 means keeping all 2 active is not allowed.
    settings = {
        "p_num_masked": [0.0, 1.0],
        "p_detector": {"H1": 0.5, "L1": 0.5},
    }
    with pytest.raises(ValueError, match="not allowing 2 active"):
        _validate_detectors_transformer(["H1", "L1"], ["H1", "L1"], settings)


def test_validate_absent_detector_never_masked_raises():
    # p_detector[H1] = 0: H1 was never masked in training, so it must be present.
    settings = {
        "p_num_masked": [0.6, 0.4],
        "p_detector": {"H1": 0.0, "L1": 1.0},
    }
    with pytest.raises(ValueError, match="never masked"):
        _validate_detectors_transformer(["L1"], ["H1", "L1"], settings)


def test_validate_present_detector_with_zero_mask_probability_allowed():
    # The always-kept detector being present is the in-distribution case.
    settings = {
        "p_num_masked": [0.6, 0.4],
        "p_detector": {"H1": 0.0, "L1": 1.0},
    }
    _validate_detectors_transformer(["H1"], ["H1", "L1"], settings)


def test_validate_missing_p_mask_keys_impose_no_constraint():
    # Absent keys mean MaskDetectors defaulted to uniform probabilities.
    _validate_detectors_transformer(["H1"], ["H1", "L1"], {})


def test_validate_more_absent_detectors_than_p_mask_allows_raises():
    # A length-1 p_num_masked allows masking 0 detectors only.
    settings = {"p_num_masked": [1.0]}
    with pytest.raises(ValueError, match="not allowing"):
        _validate_detectors_transformer(["H1"], ["H1", "L1"], settings)


def test_validate_absent_detector_missing_from_p_detector_raises():
    settings = {
        "p_num_masked": [0.6, 0.3, 0.1],
        "p_detector": {"H1": 0.5, "L1": 0.5},  # V1 missing -> treated as never masked
    }
    with pytest.raises(ValueError, match="never masked"):
        _validate_detectors_transformer(["H1", "L1"], ["H1", "L1", "V1"], settings)


# --- check_detector_update ---


def test_check_flexible_valid():
    meta = _make_metadata(["H1", "L1"], mask_detectors=HL_SETTINGS)
    check_detector_update(meta, ["H1", "L1"])  # no error


def test_check_flexible_single_detector():
    meta = _make_metadata(["H1", "L1"], mask_detectors=HL_SETTINGS)
    check_detector_update(meta, ["H1"])  # no error


def test_check_flexible_invalid_subset_raises():
    meta = _make_metadata(["H1", "L1"], mask_detectors=HL_SETTINGS)
    with pytest.raises(ValueError):
        check_detector_update(meta, ["H1", "V1"])


def test_check_mask_random_tokens_any_subset_allowed():
    # mask_random_tokens alone imposes no detector constraint.
    meta = _make_metadata(
        ["H1", "L1"], mask_random_tokens={"p_mask": 0.2, "max_num_tokens": 10}
    )
    check_detector_update(meta, ["H1"])  # no error


def test_check_no_tokenization_exact_match():
    meta = _make_metadata(["H1", "L1"])
    check_detector_update(meta, ["H1", "L1"])  # no error


def test_check_no_tokenization_mismatch_raises():
    meta = _make_metadata(["H1", "L1"])
    with pytest.raises(ValueError, match="do not match"):
        check_detector_update(meta, ["H1"])


def test_check_tokenized_without_masking_requires_exact_match():
    meta = _make_metadata(["H1", "L1"])
    meta["train_settings"]["data"]["tokenization"] = {"token_size": 16}
    with pytest.raises(ValueError, match="do not match"):
        check_detector_update(meta, ["H1"])


# --- check_frequency_updates with raw values ---


def _flexible_meta():
    meta = _make_metadata(["H1", "L1", "V1"])
    meta["train_settings"]["data"]["tokenization"] = {
        "mask_frequency_range": {
            "p_mask": 0.25,
            "f_min_upper": 180.0,
            "f_max_lower": 80.0,
            "p_same_all_detectors": 0.7,
        }
    }
    meta["dataset_settings"] = {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.25,
        }
    }
    return meta


def test_check_frequency_updates_partial_dict_over_subset():
    """A dict constrains only the detectors it names."""
    check_frequency_updates(_flexible_meta(), f_min={"H1": 30.0}, f_max=448.0)


def test_check_frequency_updates_unknown_detector_key_raises():
    with pytest.raises(ValueError, match="not.*trained with"):
        check_frequency_updates(_flexible_meta(), f_min={"K1": 30.0})


def test_check_frequency_updates_out_of_envelope_raises():
    with pytest.raises(ValueError, match="f_min_upper"):
        check_frequency_updates(_flexible_meta(), f_min={"H1": 200.0})


def test_check_frequency_updates_unchanged_value_allowed_without_flexibility():
    """dingo_pipe always passes the model values; an inflexible model must accept
    them (only a *changed* range requires flexibility)."""
    meta = _flexible_meta()
    del meta["train_settings"]["data"]["tokenization"]
    check_frequency_updates(meta, f_min=20.0, f_max=1024.0)
    with pytest.raises(ValueError, match="not trained with variable"):
        check_frequency_updates(meta, f_min=30.0)


# ---------------------------------------------------------------------------
# _validate_psd_notches / check_psd_notches
# ---------------------------------------------------------------------------

NOTCH_SETTINGS = {
    "p_per_detector": 0.3,
    "max_width": 4.0,
    "f_min": 40.0,
    "f_max": 500.0,
}


def _notch_data_settings(**tok):
    return {"detectors": DETECTORS, "tokenization": tok}


def test_psd_notches_inside_envelope_pass_silently(domain):
    settings = _notch_data_settings(mask_frequency_notches=NOTCH_SETTINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_psd_notches(
            {"H1": [59.0, 61.0], "L1": [[59.0, 61.0], [119.0, 121.0]]}, domain, settings
        )


def test_psd_notches_accept_arrays(domain):
    # Interval lists reloaded from an HDF5 file are 2-D arrays.
    settings = _notch_data_settings(mask_frequency_notches=NOTCH_SETTINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_psd_notches(
            {
                "H1": np.array([[59.0, 61.0], [119.0, 121.0]]),
                "L1": np.array([59.0, 61.0]),
            },
            domain,
            settings,
        )


def test_psd_notches_unknown_detector_raises(domain):
    settings = _notch_data_settings(mask_frequency_notches=NOTCH_SETTINGS)
    with pytest.raises(ValueError, match="not.*trained with"):
        _validate_psd_notches({"V1": [59.0, 61.0]}, domain, settings)


@pytest.mark.parametrize(
    "interval", [[20.0, 25.0], [1000.0, 1024.0], [10.0, 25.0], [61.0, 59.0]]
)
def test_psd_notches_edge_or_empty_interval_raises(domain, interval):
    """A notch touching f_min / f_max would be dropped as edge padding at data
    generation; an inverted interval is empty. Both are configuration errors."""
    settings = _notch_data_settings(mask_frequency_notches=NOTCH_SETTINGS)
    with pytest.raises(ValueError, match="touches the domain bounds|is empty"):
        _validate_psd_notches({"H1": interval}, domain, settings)


@pytest.mark.parametrize("interval", [[30.0, 32.0], [59.0, 64.5], [498.0, 502.0]])
def test_psd_notches_outside_envelope_warn(domain, interval):
    """Below the trained range, wider than max_width, above the trained range."""
    settings = _notch_data_settings(mask_frequency_notches=NOTCH_SETTINGS)
    with pytest.warns(UserWarning, match="training envelope"):
        _validate_psd_notches({"H1": interval}, domain, settings)


def test_psd_notches_without_notch_training_warn(domain):
    with pytest.warns(UserWarning, match="not trained with mask_frequency_notches"):
        _validate_psd_notches({"H1": [59.0, 61.0]}, domain, {"detectors": DETECTORS})
    settings = _notch_data_settings(
        mask_random_tokens={"p_mask": 0.4, "max_num_tokens": 10}
    )
    with pytest.warns(UserWarning, match="mask_random_tokens"):
        _validate_psd_notches({"H1": [59.0, 61.0]}, domain, settings)


def test_check_psd_notches_from_metadata():
    meta = _flexible_meta()
    meta["train_settings"]["data"]["tokenization"][
        "mask_frequency_notches"
    ] = NOTCH_SETTINGS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_psd_notches(meta, {"H1": [[59.0, 61.0]]})
    with pytest.raises(ValueError, match="touches the domain bounds"):
        check_psd_notches(meta, {"H1": [[20.0, 25.0]]})


# ---------------------------------------------------------------------------
# Per-detector bounds, and the importance-sampling-only range
# ---------------------------------------------------------------------------


def test_resolve_frequency_bounds_defaults_to_domain(domain):
    assert resolve_frequency_bounds(DETECTORS, domain) == {
        "H1": (20.0, 1024.0),
        "L1": (20.0, 1024.0),
    }


def test_resolve_frequency_bounds_expands_float_and_dict(domain):
    bounds = resolve_frequency_bounds(
        DETECTORS,
        domain,
        minimum_frequency=30,
        maximum_frequency={"H1": 512.0, "L1": 448.0},
    )
    assert bounds == {"H1": (30.0, 512.0), "L1": (30.0, 448.0)}


def test_resolve_frequency_bounds_partial_dict(domain):
    # A dict may name any of the analyzed detectors; the others keep the domain
    # bound. A detector outside the list is an error.
    bounds = resolve_frequency_bounds(
        DETECTORS, domain, maximum_frequency={"H1": 512.0}
    )
    assert bounds == {"H1": (20.0, 512.0), "L1": (20.0, 1024.0)}
    with pytest.raises(ValueError, match="not analyzed"):
        resolve_frequency_bounds(DETECTORS, domain, maximum_frequency={"V1": 512.0})


def test_frequency_bound_tokenized_network_refuses_the_crop_license(domain):
    settings = {
        "detectors": DETECTORS,
        "tokenization": {"token_size": 16},
        "random_strain_cropping": {"cropping_probability": 0.5, "f_min_upper": 60.0},
    }
    # The model values (dingo_pipe always passes them) are fine; a change is not.
    _validate_frequency_bound(20.0, "minimum_frequency", domain, settings)
    with pytest.raises(ValueError, match="mask_frequency_range"):
        _validate_frequency_bound(40.0, "minimum_frequency", domain, settings)


def test_frequency_bound_expands_over_the_analyzed_detectors(domain):
    settings = {
        "detectors": ["H1", "L1", "V1"],
        "tokenization": {"mask_frequency_range": {"p_mask": 0.5, "f_min_upper": 50.0}},
    }
    analyzed = ["H1", "L1"]
    _validate_frequency_bound(30.0, "minimum_frequency", domain, settings, analyzed)
    _validate_frequency_bound(
        {"H1": 30.0}, "minimum_frequency", domain, settings, analyzed
    )
    with pytest.raises(ValueError, match="not analyzed"):
        _validate_frequency_bound(
            {"V1": 30.0}, "minimum_frequency", domain, settings, analyzed
        )


def _model_metadata_without_cropping(domain_settings=DOMAIN_SETTINGS):
    return {
        "train_settings": {"data": {"detectors": DETECTORS}},
        "dataset_settings": {"domain": domain_settings},
    }


def test_importance_sampling_range_needs_no_cropping_license():
    # The network never sees an importance-sampling-only range, so narrowing and
    # widening are both allowed without random strain cropping (the event-time
    # check refuses to narrow).
    metadata = _model_metadata_without_cropping()
    with pytest.raises(ValueError, match="not trained with variable"):
        check_frequency_updates(metadata, f_max=512.0)
    for kwargs in (
        {"minimum_frequency": 30.0, "maximum_frequency": {"H1": 512.0, "L1": 448.0}},
        {
            "minimum_frequency": 10.0,
            "maximum_frequency": 2048.0,
            "sampling_frequency": 4096.0,
        },
        {},
    ):
        assert check_importance_sampling_frequency_range(metadata, **kwargs) is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"maximum_frequency": 2048.0, "sampling_frequency": 2048.0}, "Nyquist"),
        ({"minimum_frequency": 0.0}, "positive"),
        ({"minimum_frequency": 600.0, "maximum_frequency": 512.0}, "non-empty"),
        ({"maximum_frequency": {"V1": 512.0}}, "not analyzed"),
    ],
)
def test_importance_sampling_range_rejects_bad_requests(kwargs, match):
    with pytest.raises(ValueError, match=match):
        check_importance_sampling_frequency_range(
            _model_metadata_without_cropping(), **kwargs
        )


def test_importance_sampling_range_on_multibanded_model_is_not_limited_to_bands():
    # The base-domain likelihood of a multibanded model evaluates on a uniform
    # grid, so a wider range is allowed like any other (up to Nyquist).
    mfd_settings = {
        "type": "MultibandedFrequencyDomain",
        "nodes": [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        "delta_f_initial": 0.0625,
        "base_domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 2048.0,
            "delta_f": 0.0625,
        },
    }
    metadata = _model_metadata_without_cropping(mfd_settings)
    assert (
        check_importance_sampling_frequency_range(
            metadata, maximum_frequency=1500.0, sampling_frequency=4096.0
        )
        is None
    )
