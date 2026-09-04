"""Tests for GWSampler: frequency validation, domain construction and reference-time
correction, plus the transformer path (_initialize_transforms, _run_sampler, detector
validation)."""

import warnings

import numpy as np
import pytest
import torch
from astropy.time import Time
from astropy.utils import iers
from unittest.mock import MagicMock

from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.transforms import GetItem
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.inference.gw_samplers import (
    GWSampler,
    check_detector_update,
    check_frequency_updates,
    check_psd_notches,
    _validate_detectors_transformer,
    _validate_frequency_bound,
    _validate_psd_notches,
)
from dingo.gw.transforms import (
    StrainTokenization,
    UnpackDict,
    ToTorch,
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


# ---------------------------------------------------------------------------
# Transformer path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STANDARDIZATION = {
    "mean": {"chirp_mass": 20.0},
    "std": {"chirp_mass": 5.0},
}
INFERENCE_PARAMS = ["chirp_mass"]


def _make_domain(f_min=20.0, f_max=128.0, delta_f=0.25):
    return UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=delta_f)


def _make_sampler_stub(domain, tokenization_settings=None):
    """Return a GWSampler with the minimum attributes set to call _initialize_transforms.

    Uses object.__setattr__ to bypass Sampler.__init__, so no real model or dataset
    is needed.
    """
    data_settings = {
        "detectors": DETECTORS,
        "standardization": STANDARDIZATION,
        "ref_time": 1126259462.391,
    }
    if tokenization_settings is not None:
        data_settings["tokenization"] = tokenization_settings

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    sampler = object.__new__(GWSampler)
    sampler.domain = domain
    sampler.model = mock_model
    metadata = {
        "train_settings": {"data": data_settings},
        "dataset_settings": {"intrinsic_prior": {}},
    }
    sampler.metadata = metadata
    # GWSamplerMixin.detectors reads from base_model_metadata (== metadata for non-GNPE).
    sampler.base_model_metadata = metadata
    sampler.inference_parameters = INFERENCE_PARAMS
    sampler._minimum_frequency = None
    sampler._maximum_frequency = None
    sampler._detectors = None
    return sampler


def _make_context(domain, rng=None):
    """Build a minimal {'waveform': ..., 'asds': ...} dict for *domain*."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(domain.sample_frequencies)
    return {
        "waveform": {
            d: (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
                np.complex64
            )
            for d in DETECTORS
        },
        "asds": {d: rng.uniform(1e-24, 1e-23, n).astype(np.float32) for d in DETECTORS},
    }


# ---------------------------------------------------------------------------
# _initialize_transforms — resnet (no tokenization)
# ---------------------------------------------------------------------------


def test_resnet_path_uses_get_item():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=None)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    assert isinstance(transforms[-1], GetItem)
    assert transforms[-1].key == "waveform"


def test_resnet_path_has_no_strain_tokenization():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=None)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert StrainTokenization not in types
    assert UnpackDict not in types


def test_resnet_path_output_is_tensor():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain)
    sampler._initialize_transforms()

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    assert isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# _initialize_transforms — transformer (with tokenization)
# ---------------------------------------------------------------------------

TOK_SETTINGS = {
    "token_size": 16,
    "num_tokens_per_block": None,
    "drop_last_token": False,
}


def test_transformer_path_has_strain_tokenization():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert StrainTokenization in types


def test_transformer_path_has_unpack_dict():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert UnpackDict in types


def test_transformer_path_no_get_item():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    types = [type(t) for t in sampler.transform_pre.transforms]
    assert GetItem not in types


def test_transformer_tokenization_precedes_to_torch():
    """StrainTokenization must come before ToTorch in the chain."""
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    indices = {type(t): i for i, t in enumerate(transforms)}
    assert indices[StrainTokenization] < indices[ToTorch]


def test_transformer_unpack_dict_follows_to_torch():
    """UnpackDict must come after ToTorch in the chain."""
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    transforms = sampler.transform_pre.transforms
    indices = {type(t): i for i, t in enumerate(transforms)}
    assert indices[UnpackDict] > indices[ToTorch]


def test_transformer_path_output_is_list_of_three_tensors():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    context = _make_context(domain)
    x = sampler.transform_pre(context)
    assert isinstance(x, list)
    assert len(x) == 3
    waveform, position, mask = x
    assert isinstance(waveform, torch.Tensor)
    assert isinstance(position, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool


def test_transformer_path_waveform_and_position_num_tokens_match():
    domain = _make_domain()
    sampler = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    sampler._initialize_transforms()

    context = _make_context(domain)
    waveform, position, mask = sampler.transform_pre(context)
    assert waveform.shape[0] == position.shape[0] == mask.shape[0]


# ---------------------------------------------------------------------------
# _run_sampler list-handling logic
# ---------------------------------------------------------------------------


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
    "p_mask_012_detectors": [0.6, 0.3, 0.1],
    "p_mask_hlv": {"H1": 0.3, "L1": 0.3, "V1": 0.4},
}

HL_SETTINGS = {
    "p_mask_012_detectors": [0.6, 0.4],
    "p_mask_hlv": {"H1": 0.5, "L1": 0.5},
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
    # p_mask_012_detectors[0] = 0 means keeping all 2 active is not allowed.
    settings = {
        "p_mask_012_detectors": [0.0, 1.0],
        "p_mask_hlv": {"H1": 0.5, "L1": 0.5},
    }
    with pytest.raises(ValueError, match="not allowing 2 active"):
        _validate_detectors_transformer(["H1", "L1"], ["H1", "L1"], settings)


def test_validate_absent_detector_never_masked_raises():
    # p_mask_hlv[H1] = 0: H1 was never masked in training, so it must be present.
    settings = {
        "p_mask_012_detectors": [0.6, 0.4],
        "p_mask_hlv": {"H1": 0.0, "L1": 1.0},
    }
    with pytest.raises(ValueError, match="never masked"):
        _validate_detectors_transformer(["L1"], ["H1", "L1"], settings)


def test_validate_present_detector_with_zero_mask_probability_allowed():
    # The always-kept detector being present is the in-distribution case.
    settings = {
        "p_mask_012_detectors": [0.6, 0.4],
        "p_mask_hlv": {"H1": 0.0, "L1": 1.0},
    }
    _validate_detectors_transformer(["H1"], ["H1", "L1"], settings)


def test_validate_missing_p_mask_keys_impose_no_constraint():
    # Absent keys mean MaskDetectors defaulted to uniform probabilities.
    _validate_detectors_transformer(["H1"], ["H1", "L1"], {})


def test_validate_more_absent_detectors_than_p_mask_allows_raises():
    # A length-1 p_mask_012_detectors allows masking 0 detectors only.
    settings = {"p_mask_012_detectors": [1.0]}
    with pytest.raises(ValueError, match="not allowing"):
        _validate_detectors_transformer(["H1"], ["H1", "L1"], settings)


def test_validate_absent_detector_missing_from_p_mask_hlv_raises():
    settings = {
        "p_mask_012_detectors": [0.6, 0.3, 0.1],
        "p_mask_hlv": {"H1": 0.5, "L1": 0.5},  # V1 missing -> treated as never masked
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


# ---------------------------------------------------------------------------
# _run_sampler list-handling logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_list", [True, False])
def test_run_sampler_adds_and_removes_batch_dimension(gw_sampler, monkeypatch, as_list):
    """_run_sampler hands the network batched inputs (a list of tensors from a
    tokenized chain, a single tensor from a resnet chain) and squeezes the batch
    dimension back out of the samples."""
    tensors = [
        torch.randn(86, 48),
        torch.randn(86, 3),
        torch.zeros(86, dtype=torch.bool),
    ]
    monkeypatch.setattr(
        gw_sampler, "transform_pre", lambda ctx: tensors if as_list else tensors[0]
    )
    seen = []

    def fake_sample_and_log_prob(*x, num_samples):
        seen.extend(t.shape for t in x)
        dim = len(INFERENCE_PARAMETERS)
        return torch.zeros(1, num_samples, dim), torch.zeros(1, num_samples)

    monkeypatch.setattr(
        gw_sampler.model, "sample_and_log_prob", fake_sample_and_log_prob
    )
    result = gw_sampler._run_sampler(num_samples=5, context={"waveform": {}})
    expected = [(1, 86, 48), (1, 86, 3), (1, 86)] if as_list else [(1, 86, 48)]
    assert seen == expected
    assert len(result["log_prob"]) == 5


def test_check_tokenized_without_masking_requires_exact_match():
    meta = _make_metadata(["H1", "L1"])
    meta["train_settings"]["data"]["tokenization"] = {"token_size": 16}
    with pytest.raises(ValueError, match="do not match"):
        check_detector_update(meta, ["H1"])


def test_event_metadata_round_trips_detectors(gw_sampler):
    gw_sampler.event_metadata = {"time_event": 0.0, "detectors": ["H1", "L1"]}
    assert gw_sampler.event_metadata["detectors"] == ["H1", "L1"]


def test_detectors_setter_validates_against_model(gw_sampler):
    # Non-tokenized model: a detector subset must be rejected in the setter itself,
    # so the API path is guarded, not only dingo_pipe.
    with pytest.raises(ValueError, match="do not match"):
        gw_sampler.detectors = ["H1"]


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


def test_event_metadata_round_trips_psd_notch_dict(gw_sampler):
    # The fixture model has no notch training: the setter warns but proceeds.
    with pytest.warns(UserWarning, match="mask_frequency_notches"):
        gw_sampler.event_metadata = {
            "time_event": 0.0,
            "psd_notch_dict": {"H1": [[60.0, 61.0]]},
        }
    assert gw_sampler.event_metadata["psd_notch_dict"] == {"H1": [[60.0, 61.0]]}


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


def test_event_metadata_rebuilds_transform_chain_once(gw_sampler, monkeypatch):
    """Assigning event_metadata applies every setting but rebuilds the chain once;
    a property assigned by hand still rebuilds immediately."""
    calls = []
    monkeypatch.setattr(gw_sampler, "_initialize_transforms", lambda: calls.append(1))
    gw_sampler.event_metadata = {
        "time_event": 0.0,
        "detectors": DETECTORS,
        "minimum_frequency": DOMAIN_SETTINGS["f_min"],
        "maximum_frequency": DOMAIN_SETTINGS["f_max"],
    }
    assert len(calls) == 1
    gw_sampler.minimum_frequency = DOMAIN_SETTINGS["f_min"]
    assert len(calls) == 2


def test_event_metadata_omits_psd_notch_dict_when_absent(gw_sampler):
    gw_sampler.event_metadata = {"time_event": 0.0}
    assert "psd_notch_dict" not in gw_sampler.event_metadata


def test_frequency_update_chain_drops_bin_masking_for_tokenized():
    """For tokenized models the bin-level zeroing is inert (every unmasked token
    lies fully inside the requested range), so it is not built into the chain."""
    domain = _make_domain()
    stub = _make_sampler_stub(domain, tokenization_settings=TOK_SETTINGS)
    stub._minimum_frequency = 30.0
    stub._initialize_transforms()
    names = [type(t).__name__ for t in stub.transform_pre.transforms]
    assert "MaskDataForFrequencyRangeUpdate" not in names
    assert "MaskTokensForFrequencyRangeUpdate" in names


def test_frequency_update_chain_keeps_bin_masking_for_resnet():
    domain = _make_domain()
    stub = _make_sampler_stub(domain)
    stub._minimum_frequency = 30.0
    stub._initialize_transforms()
    names = [type(t).__name__ for t in stub.transform_pre.transforms]
    assert "MaskDataForFrequencyRangeUpdate" in names
