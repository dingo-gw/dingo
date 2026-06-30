import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.gwutils import (
    get_extrinsic_prior_dict,
    get_mismatch,
    get_standardization_dict,
    get_window,
)


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
