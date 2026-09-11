import warnings

import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.gwutils import (
    detect_asd_notches,
    get_extrinsic_prior_dict,
    get_mismatch,
    get_standardization_dict,
    get_window,
)
from dingo.gw.noise.asd_dataset import HIGH_ASD_VALUE


# ---------------------------------------------------------------------------
# get_mismatch
#
# mismatch = 1 - overlap, with overlap = <a, b> / sqrt(<a, a> <b, b>).
# Properties checked here mirror bilby's overlap test
# (bilby/test/gw/utils_test.py::TestGWUtils::test_overlap), adapted to dingo's
# get_mismatch.
# ---------------------------------------------------------------------------


@pytest.fixture()
def domain():
    return UniformFrequencyDomain(20.0, 256.0, delta_f=0.5)


@pytest.fixture()
def waveforms(domain):
    rng = np.random.default_rng(0)
    n = len(domain)
    a = rng.normal(size=n) + 1j * rng.normal(size=n)
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    return a, b


def test_mismatch_of_identical_waveforms_is_zero(domain, waveforms):
    a, _ = waveforms
    assert get_mismatch(a, a, domain) == pytest.approx(0.0, abs=1e-12)


def test_mismatch_is_scale_invariant(domain, waveforms):
    # Overlap is normalized, so a rescaling of one waveform leaves the mismatch at 0.
    a, _ = waveforms
    assert get_mismatch(a, 3.0 * a, domain) == pytest.approx(0.0, abs=1e-12)


def test_mismatch_is_symmetric(domain, waveforms):
    a, b = waveforms
    assert get_mismatch(a, b, domain) == pytest.approx(get_mismatch(b, a, domain))


def test_mismatch_is_in_valid_range(domain, waveforms):
    # overlap in [-1, 1]  =>  mismatch = 1 - overlap in [0, 2].
    a, b = waveforms
    assert 0.0 <= get_mismatch(a, b, domain) <= 2.0


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------


def test_get_window_tukey_length_and_range():
    T, f_s = 4.0, 1024
    window = get_window({"type": "tukey", "roll_off": 0.4, "T": T, "f_s": f_s})
    assert len(window) == int(T * f_s)
    assert np.all((window >= 0.0) & (window <= 1.0))
    # A Tukey window tapers to (near) zero at the edges.
    assert window[0] < 1e-6 and window[-1] < 1e-6


def test_get_window_unknown_type_raises():
    with pytest.raises(NotImplementedError, match="window type"):
        get_window({"type": "not_a_window"})


# ---------------------------------------------------------------------------
# get_extrinsic_prior_dict
# ---------------------------------------------------------------------------


def test_get_extrinsic_prior_dict_expands_default_and_keeps_override():
    override = "bilby.core.prior.Uniform(minimum=100, maximum=1000)"
    out = get_extrinsic_prior_dict({"ra": "default", "luminosity_distance": override})
    # "default" is replaced by the package default prior (no longer the literal string).
    assert out["ra"] != "default"
    # A non-default value is passed through unchanged.
    assert out["luminosity_distance"] == override


# ---------------------------------------------------------------------------
# get_standardization_dict
# ---------------------------------------------------------------------------


class _StubWaveformDataset:
    """Minimal stand-in exposing only what get_standardization_dict needs:
    parameter_mean_std() for intrinsic params (extrinsic ones come from the prior)."""

    def __init__(self, luminosity_distance_std=0.0):
        self._ld_std = luminosity_distance_std
        self.parameters = pd.DataFrame({"chirp_mass": [30.0]})

    def parameter_mean_std(self):
        mean = {"chirp_mass": 30.0, "luminosity_distance": 100.0}
        std = {"chirp_mass": 5.0, "luminosity_distance": self._ld_std}
        return mean, std


@pytest.fixture()
def extrinsic_prior():
    return get_extrinsic_prior_dict(
        {
            "ra": "default",
            "dec": "default",
            "psi": "default",
            "luminosity_distance": (
                "bilby.core.prior.Uniform("
                "minimum=100, maximum=1000, name='luminosity_distance')"
            ),
            "geocent_time": (
                "bilby.core.prior.Uniform("
                "minimum=-0.1, maximum=0.1, name='geocent_time')"
            ),
        }
    )


def test_get_standardization_dict_combines_intrinsic_and_extrinsic(extrinsic_prior):
    selected = ["chirp_mass", "ra", "luminosity_distance"]
    out = get_standardization_dict(extrinsic_prior, _StubWaveformDataset(), selected)

    assert set(out["mean"]) == set(selected) == set(out["std"])
    # Intrinsic parameter values come straight from the dataset.
    assert out["mean"]["chirp_mass"] == 30.0
    assert out["std"]["chirp_mass"] == 5.0
    # Extrinsic parameter standardization is analytic / from the prior.
    assert out["std"]["ra"] > 0


def test_get_standardization_dict_rejects_nonzero_intrinsic_std_for_extrinsic(
    extrinsic_prior,
):
    # luminosity_distance is sampled as an extrinsic parameter, so the dataset must
    # hold it at a fixed (std 0) value; a non-zero intrinsic std is an error.
    wfd = _StubWaveformDataset(luminosity_distance_std=5.0)
    with pytest.raises(ValueError, match="fixed value"):
        get_standardization_dict(extrinsic_prior, wfd, ["chirp_mass"])


# ---------------------------------------------------------------------------
# detect_asd_notches
# ---------------------------------------------------------------------------


def _make_asd_array(domain, notch_intervals=None):
    """Build a full-length ASD array (length max_idx + 1) with real-valued noise
    below HIGH_ASD_VALUE, except inside notch_intervals where ASD = HIGH_ASD_VALUE.
    Edge-padding bins (0 .. min_idx-1) are set to HIGH_ASD_VALUE by convention.
    """
    n = domain.max_idx + 1
    asd = np.full(n, 1e-23)  # realistic noise amplitude
    # edge padding
    asd[: domain.min_idx] = HIGH_ASD_VALUE
    if notch_intervals:
        freqs = domain.sample_frequencies
        for f_lo, f_hi in notch_intervals:
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            asd[mask] = HIGH_ASD_VALUE
    return asd


def test_detect_asd_notches_no_notch():
    """No notches in any detector → returns None."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.125)
    asd = _make_asd_array(domain)
    result = detect_asd_notches({"H1": asd, "L1": asd}, domain)
    assert result is None


def test_detect_asd_notches_single_notch():
    """Single interior notch is correctly detected."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.125)
    f_lo, f_hi = 59.0, 61.0
    asd = _make_asd_array(domain, notch_intervals=[[f_lo, f_hi]])
    result = detect_asd_notches({"H1": asd}, domain)
    assert result is not None
    assert "H1" in result
    intervals = result["H1"]
    assert len(intervals) == 1
    detected_lo, detected_hi = intervals[0]
    assert detected_lo >= f_lo
    assert detected_hi <= f_hi + domain.delta_f


def test_detect_asd_notches_multiple_notches():
    """Multiple disjoint notches per detector are all detected."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.125)
    notches = [[59.0, 61.0], [119.0, 121.0]]
    asd = _make_asd_array(domain, notch_intervals=notches)
    result = detect_asd_notches({"H1": asd}, domain)
    assert result is not None
    assert len(result["H1"]) == 2


def test_detect_asd_notches_edge_padding_ignored():
    """Edge-padding at f_min (index 0 of valid band) is not reported as a notch."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.125)
    # Set the very first valid bin (at f_min) to HIGH_ASD_VALUE to mimic edge padding.
    asd = _make_asd_array(domain)
    asd[domain.min_idx] = HIGH_ASD_VALUE
    result = detect_asd_notches({"H1": asd}, domain)
    assert result is None


def test_detect_asd_notches_edge_padding_ignored_at_f_max():
    """A high run touching f_max (bilby fills frequencies beyond a PSD file with inf)
    is edge padding, not a notch. One bin is silent; wider warns that the PSD does
    not cover the model band."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.125)
    asd = _make_asd_array(domain)
    asd[-1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect_asd_notches({"H1": asd}, domain) is None
    asd[-3:] = np.inf
    with pytest.warns(UserWarning, match="does not cover"):
        assert detect_asd_notches({"H1": asd}, domain) is None


def test_detect_asd_notches_per_detector():
    """Notch in H1 only is not reported for L1."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.125)
    asd_h1 = _make_asd_array(domain, notch_intervals=[[59.0, 61.0]])
    asd_l1 = _make_asd_array(domain)
    result = detect_asd_notches({"H1": asd_h1, "L1": asd_l1}, domain)
    assert result is not None
    assert "H1" in result
    assert "L1" not in result


def test_detect_asd_notches_multibanded_uses_base_domain():
    """Stored ASDs live on the base grid; MFD indices must not be used (this
    returned wrong notch frequencies before)."""
    base = UniformFrequencyDomain(f_min=20.0, f_max=100.0, delta_f=0.25)
    mfd = MultibandedFrequencyDomain(
        nodes=[20.0, 36.0, 100.0], delta_f_initial=0.25, base_domain=base
    )
    asd = np.full(len(base), 1e-23)
    freqs = base.sample_frequencies
    asd[(freqs >= 60.0) & (freqs <= 61.0)] = HIGH_ASD_VALUE
    notches = detect_asd_notches({"H1": asd}, mfd)
    assert notches == {"H1": [[60.0, 61.0]]}
