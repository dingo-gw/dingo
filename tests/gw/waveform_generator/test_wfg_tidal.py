"""Tests for tidal (BNS) waveform generation with NRTidal approximants."""

import lalsimulation as LS
import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator, sum_contributions_m

APPROXIMANT = "IMRPhenomXP_NRTidalv3"
F_REF = 30.0

BNS_PARAMETERS = {
    "chirp_mass": 1.4,
    "mass_ratio": 0.8,
    "a_1": 0.3,
    "a_2": 0.2,
    "tilt_1": 1.0,
    "tilt_2": 0.5,
    "phi_12": 0.7,
    "phi_jl": 0.4,
    "theta_jn": 1.2,
    "phase": 0.6,
    # Not 1.0, which would trigger the PhenomX multibanding-threshold retry path.
    "luminosity_distance": 100.0,
    "lambda_1": 400.0,
    "lambda_2": 300.0,
}


@pytest.fixture
def ufd():
    return UniformFrequencyDomain(f_min=30.0, f_max=1024.0, delta_f=0.25)


@pytest.fixture
def mfd():
    domain_settings = {
        "nodes": [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0],
        "delta_f_initial": 0.0625,
        "base_domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1037.9375,
            "delta_f": 0.0625,
        },
    }
    return MultibandedFrequencyDomain(**domain_settings)


@pytest.fixture(params=["ufd", "mfd"])
def domain(request):
    return request.getfixturevalue(request.param)


def test_tidal_waveform(domain):
    """Polarizations can be generated with tidal parameters, are consistent with the
    domain, and the tidal parameters have an effect on the waveform."""
    wf_gen = WaveformGenerator(APPROXIMANT, domain, F_REF, spin_conversion_phase=0.0)
    hp_tidal = wf_gen.generate_hplus_hcross(BNS_PARAMETERS)["h_plus"]

    assert len(hp_tidal) == len(domain)
    assert np.all(np.isfinite(hp_tidal))
    assert np.max(np.abs(hp_tidal)) > 0.0

    # Regression guard against silently dropped tides: zero deformabilities must
    # change the waveform (measured effect for these parameters is ~10%).
    hp_zero = wf_gen.generate_hplus_hcross(
        {**BNS_PARAMETERS, "lambda_1": 0.0, "lambda_2": 0.0}
    )["h_plus"]
    assert np.max(np.abs(hp_tidal - hp_zero)) / np.max(np.abs(hp_zero)) > 1e-3


def test_tidal_modes_raise(ufd):
    """Without an explicit mode_list, mode-separated generation must fail with an
    informative error: the DFT path cannot size its grid (deliberately no
    DEFAULT_ELL_MAX entry) and falls back to the individual-mode path, for which
    LALSimulation implements no FD modes; the message points at the DFT path."""
    wf_gen = WaveformGenerator(APPROXIMANT, ufd, F_REF, spin_conversion_phase=0.0)
    with pytest.warns(UserWarning, match="Falling back to the individual-mode"):
        with pytest.raises(NotImplementedError, match="use_dft_phase_decomposition"):
            wf_gen.generate_hplus_hcross_m(BNS_PARAMETERS)


def test_tidal_mode_decomposition(ufd):
    """With an explicit mode_list, generate_hplus_hcross_m runs through the DFT
    phase decomposition (ell_max=2) and reproduces direct generation at an
    off-grid phase shift to round-off. (There is deliberately no DEFAULT_ELL_MAX
    entry: for these models the (2, 2) synthetic phase is exact in Bilby's spin
    convention, so the mode decomposition is opt-in.)"""
    wf_gen = WaveformGenerator(
        APPROXIMANT,
        ufd,
        F_REF,
        mode_list=[(2, 2), (2, -2)],
        spin_conversion_phase=0.0,
    )
    pol_m = wf_gen.generate_hplus_hcross_m(BNS_PARAMETERS)
    assert sorted(pol_m) == [-2, -1, 0, 1, 2]

    delta = 0.83  # not a multiple of 2 pi / 5, so aliasing would show up here
    h_rec = sum_contributions_m(pol_m, phase_shift=delta)
    h_true = wf_gen.generate_hplus_hcross(
        {**BNS_PARAMETERS, "phase": BNS_PARAMETERS["phase"] + delta}
    )
    for pol in ("h_plus", "h_cross"):
        err = np.max(np.abs(h_true[pol] - h_rec[pol])) / np.max(np.abs(h_true[pol]))
        assert err < 1e-12


def test_tidal_22_approximation_mismatch(ufd):
    """Document the accuracy of the approximation_22_mode shortcut for this
    precessing BNS configuration: it assumes h(phase) = h(0) exp(2i phase), but
    precession spreads the co-precessing (2, +-2) modes over inertial-frame
    m in [-2, 2]. Measured flat-noise mismatches on this domain are 3e-4 - 3e-2,
    while the DFT mode decomposition is exact to round-off."""
    wf_gen = WaveformGenerator(
        APPROXIMANT,
        ufd,
        F_REF,
        mode_list=[(2, 2), (2, -2)],
        spin_conversion_phase=0.0,
    )
    pol_m = wf_gen.generate_hplus_hcross_m(BNS_PARAMETERS)
    h0 = wf_gen.generate_hplus_hcross(BNS_PARAMETERS)
    for delta in (0.83, 2.1):
        h_true = wf_gen.generate_hplus_hcross(
            {**BNS_PARAMETERS, "phase": BNS_PARAMETERS["phase"] + delta}
        )
        h_dft = sum_contributions_m(pol_m, phase_shift=delta)
        for pol in ("h_plus", "h_cross"):
            mm_22 = get_mismatch(h_true[pol], h0[pol] * np.exp(2j * delta), ufd)
            mm_dft = get_mismatch(h_true[pol], h_dft[pol], ufd)
            assert mm_22 > 1e-4
            assert mm_dft < 1e-13


def test_tidal_lal_params_not_mutated(ufd):
    """Tidal insertion must not leak into the generator's shared mode-array dict."""
    wf_gen = WaveformGenerator(
        APPROXIMANT, ufd, F_REF, mode_list=[(2, 2)], spin_conversion_phase=0.0
    )
    wf_gen.generate_hplus_hcross(BNS_PARAMETERS)
    assert LS.SimInspiralWaveformParamsLookupTidalLambda1(wf_gen.lal_params) == 0.0


def test_tidal_prior_defaults():
    """The `default` priors for lambda_1/lambda_2 expand to Uniform(0, 5000)."""
    np.random.seed(42)
    prior = build_prior_with_defaults({"lambda_1": "default", "lambda_2": "default"})
    sample = prior.sample()
    for k in ("lambda_1", "lambda_2"):
        assert 0.0 <= sample[k] <= 5000.0


def test_mfd_time_origin_grid_independent():
    """Some models (e.g. IMRPhenomXP_NRTidalv3) place the waveform in time based
    on the last frequency they are asked for: without the f_max padding, grids
    ending at 255 and 300 Hz disagree by a 64 ms time shift for this
    configuration. With the padding, evaluations on differently truncated grids
    must agree."""

    def single_band(f_end):
        return MultibandedFrequencyDomain(
            nodes=[20.0, f_end],
            delta_f_initial=1.0,
            base_domain={
                "type": "UniformFrequencyDomain",
                "f_min": 20.0,
                "f_max": 2048.0,
                "delta_f": 1.0,
            },
        )

    bns = {**BNS_PARAMETERS, "chirp_mass": 1.1975}
    h = []
    for f_end in (256.0, 301.0):
        wf_gen = WaveformGenerator(
            APPROXIMANT, single_band(f_end), F_REF, spin_conversion_phase=0.0
        )
        h.append(wf_gen.generate_hplus_hcross(bns)["h_plus"])
    n = len(h[0])
    err = np.max(np.abs(h[0] - h[1][:n])) / np.max(np.abs(h[0]))
    assert err < 1e-10


ALIGNED_PARAMETERS = {
    "chirp_mass": 30.0,
    "mass_ratio": 0.8,
    "chi_1": 0.3,
    "chi_2": -0.2,
    "theta_jn": 1.2,
    "phase": 0.6,
    "luminosity_distance": 1000.0,
}


@pytest.mark.parametrize(
    "approximant, spin_conversion_phase, exact",
    [
        (APPROXIMANT, None, True),
        # Bilby's convention is what makes it exact: with a fixed one, a phase shift
        # leaves the in-plane spins behind and the waveform is not just rescaled.
        (APPROXIMANT, 0.0, False),
        # A single (2, |m| = 2) pair is the other half: higher modes break it.
        ("IMRPhenomXPHM", None, False),
        # Aligned spins: no in-plane spins to rotate, so the convention is irrelevant.
        ("IMRPhenomD", 0.0, True),
        ("IMRPhenomXHM", 0.0, False),
    ],
)
def test_phase_is_global_factor_matches_the_waveform(
    ufd, approximant, spin_conversion_phase, exact
):
    """The declared property and a probe of the waveform itself agree on whether a
    phase shift multiplies the waveform by exp(2i phase) -- the condition for the
    (2, 2) synthetic phase, and its cached log likelihood, to be exact."""
    from dingo.gw.inference.steps import _phase_global_factor_mismatch

    if approximant in ("IMRPhenomD", "IMRPhenomXHM"):
        parameters = ALIGNED_PARAMETERS
    elif "NRTidal" in approximant:
        parameters = BNS_PARAMETERS
    else:
        parameters = {
            **{k: v for k, v in BNS_PARAMETERS.items() if not k.startswith("lambda")},
            "chirp_mass": 30.0,
        }
    wf_gen = WaveformGenerator(
        approximant, ufd, F_REF, spin_conversion_phase=spin_conversion_phase
    )
    assert wf_gen.phase_is_global_factor is exact

    mismatch = _phase_global_factor_mismatch(wf_gen, parameters)
    assert bool(mismatch < 1e-12) is exact, f"probe mismatch {mismatch:.2e}"


BNS_ALIGNED_PARAMETERS = {
    **{k: v for k, v in BNS_PARAMETERS.items() if k not in ("tilt_1", "tilt_2")},
    "tilt_1": 0.0,
    "tilt_2": 0.0,
}
NSBH_PARAMETERS = {
    **BNS_ALIGNED_PARAMETERS,
    "chirp_mass": 2.8,
    "mass_ratio": 0.2,
    "lambda_1": 0.0,
    "lambda_2": 500.0,
}


@pytest.mark.parametrize(
    "approximant, parameters, exact",
    [
        ("SEOBNRv5_ROM", ALIGNED_PARAMETERS, True),
        ("SEOBNRv5_ROM_NRTidalv3", BNS_ALIGNED_PARAMETERS, True),
        ("SEOBNRv4_ROM_NRTidalv2", BNS_ALIGNED_PARAMETERS, True),
        ("SEOBNRv4_ROM_NRTidalv2_NSBH", NSBH_PARAMETERS, True),
        ("IMRPhenomNSBH", NSBH_PARAMETERS, True),
        ("IMRPhenomXAS_NRTidalv3", BNS_ALIGNED_PARAMETERS, True),
        ("IMRPhenomD_NRTidalv2", BNS_ALIGNED_PARAMETERS, True),
        # Higher modes: generates, but a phase shift is not a global factor.
        ("SEOBNRv5HM_ROM", ALIGNED_PARAMETERS, False),
    ],
)
def test_aligned_models_generate_and_match_their_exactness_flag(
    ufd, approximant, parameters, exact
):
    """Matter and aligned-spin models run through the generator and agree with what
    `phase_is_global_factor` says about them: a finite, non-trivial waveform, and a
    probe of the waveform that confirms (or denies) the global exp(2i phase) factor.
    The SEOBNRv5 ROMs reach LALSimulation only because the generator no longer treats
    every "SEOBNRv5" name as gwsignal-only."""
    from dingo.gw.inference.steps import _phase_global_factor_mismatch

    wf_gen = WaveformGenerator(approximant, ufd, F_REF, spin_conversion_phase=0.0)
    h = wf_gen.generate_hplus_hcross(parameters)
    assert np.all(np.isfinite(h["h_plus"])) and np.all(np.isfinite(h["h_cross"]))
    assert np.max(np.abs(h["h_plus"])) > 0.0

    assert wf_gen.phase_is_global_factor is exact
    mismatch = _phase_global_factor_mismatch(wf_gen, parameters)
    assert bool(mismatch < 1e-12) is exact, f"probe mismatch {mismatch:.2e}"
