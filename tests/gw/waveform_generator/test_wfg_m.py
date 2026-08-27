"""
This tests the method WaveformGenerator.generate_hplus_hcross_m, that returns the
polarzations disentangled into contributions m \in [-l_max, ...,0, ...,l_max],
that transform as exp(-1j * m * phase_shift) under phase shifts. This is important when
treating the phase parameter as an extrinsic parameter.

Note: this only accounts for the modified argument in the spherical harmonics, not for
the rotation of phase_shift of the cartesian spins in xy plane. Our workaround is to
set wfg.spin_conversion_phase = 0.0, which sets a constant phase 0 when converting PE
spins to cartesian spins. This means that phi_12 and phi_jl have different definitions,
which needs to be accounted for in postprocessing. The tests below all use
wfg.spin_conversion_phase = 0.0.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt

from dingo.gw.waveform_generator import (
    WaveformGenerator,
    sum_contributions_m,
    NewInterfaceWaveformGenerator,
)
from dingo.gw.waveform_generator.waveform_generator import DEFAULT_ELL_MAX
from dingo.gw.gwutils import get_mismatch
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults


@pytest.fixture
def uniform_fd_domain():
    domain_settings = {
        "type": "UniformFrequencyDomain",
        "f_min": 10.0,
        "f_max": 2048.0,  # Note that if this isn't a power of 2, mismatches are worse.
        "delta_f": 0.125,
    }
    domain = build_domain(domain_settings)
    return domain


@pytest.fixture(params=["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM"])
def approximant(request):
    return request.param


@pytest.fixture
def intrinsic_prior(approximant):
    if "PHM" in approximant:
        intrinsic_dict = {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "geocent_time": 0.0,
        }
    else:
        # Aligned spins
        intrinsic_dict = {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "chi_1": 'bilby.gw.prior.AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99))',
            "chi_2": 'bilby.gw.prior.AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99))',
            "geocent_time": 0.0,
        }
    prior = build_prior_with_defaults(intrinsic_dict)
    return prior


@pytest.fixture
def wfg(uniform_fd_domain, approximant):
    if approximant in ["SEOBNRv5PHM", "SEOBNRv5HM"]:
        wfg_class = NewInterfaceWaveformGenerator
    else:
        wfg_class = WaveformGenerator
    return wfg_class(
        approximant=approximant,
        domain=uniform_fd_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )


@pytest.fixture
def num_evaluations(approximant):
    if "Phenom" in approximant:
        return 10
    elif approximant == "SEOBNRv4PHM":
        return 1
    else:
        return 10


@pytest.fixture
def tolerances(approximant):
    # Return (max, median) mismatches expected.
    if approximant == "IMRPhenomXPHM":
        # The mismatches are typically be of order 1e-5 to 1e-9. This comes from the
        # calculation of the magnitude of the orbital angular momentum, which we calculate
        # to a different order the IMRPhenomXPHM. It's tricky to get this exactly right,
        # since there are many different methods for this. But the small mismatches we do
        # get should not have a big effect in practice.
        return 2e-2, 1e-5

    elif approximant == "SEOBNRv4PHM":
        # The mismatches are typically be of order 1e-5. This is exclusively due to
        # different tapering. The reference polarizations are tapered and FFTed on the
        # level of polarizations, while for generate_hplus_hcross_m, the tapering and FFT
        # happens on the level of complex modes.
        # We tested the mismatches for 20k waveforms, and the largest mismatch encountered
        # was 7e-4, while almost all mismatches were of order 1e-5.
        return 5e-4, 5e-4

    elif approximant in ["SEOBNRv5PHM", "SEOBNRv5HM"]:
        # Tested on 1000 mismatches.
        return 1e-9, 1e-12

    else:
        return 1e-5, 1e-5


# Uncomment to test only one approximant.
try:
    import pyseobnr

    approximant_list = ["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM", "SEOBNRv5HM"]
except ImportError:
    approximant_list = ["IMRPhenomXPHM", "SEOBNRv4PHM"]


@pytest.mark.parametrize("approximant", approximant_list)
def test_generate_hplus_hcross_m(intrinsic_prior, wfg, num_evaluations, tolerances):
    mismatches = []
    for idx in range(num_evaluations):
        p = intrinsic_prior.sample()
        phase_shift = np.random.uniform(high=2 * np.pi)

        pol_m = wfg.generate_hplus_hcross_m(p)
        pol = sum_contributions_m(pol_m, phase_shift=phase_shift)
        pol_ref = wfg.generate_hplus_hcross({**p, "phase": p["phase"] + phase_shift})

        mismatches.append(
            [
                get_mismatch(
                    pol[pol_name],
                    pol_ref[pol_name],
                    wfg.domain,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                for pol_name in pol
            ]
        )

        debug = False
        if debug:
            maxval = max(mismatches[-1])
            idx = mismatches[-1].index(maxval)
            p = list(pol.keys())[idx]
            plt.figure(figsize=(10, 7))
            plt.plot(wfg.domain.sample_frequencies, pol[p], label="reconstructed")
            plt.plot(
                wfg.domain.sample_frequencies, pol_ref[p], label="ref", linestyle="--"
            )
            plt.plot(wfg.domain.sample_frequencies, pol_ref[p] - pol[p], label="diff")
            plt.legend()
            plt.xscale("log")
            plt.xlim((5, 128))
            plt.title(f"{p}, mismatch={maxval}")
            plt.show()

    mismatches = np.array(mismatches)

    assert np.max(mismatches) < tolerances[0]
    assert np.median(mismatches) < tolerances[1]


# ---------------------------------------------------------------------------
# DFT phase decomposition (use_dft_phase_decomposition=True)
#
# Instead of building the individual inertial-frame modes, the m-components are
# recovered from N = 2 * ell_max + 1 evaluations of the summed polarizations via a
# DFT. Parameters are fixed rather than drawn from the prior, so the tolerances
# below can be tight and the tests cannot fail intermittently.
# ---------------------------------------------------------------------------

dft_approximant_list = [
    a for a in approximant_list if a in ("IMRPhenomXPHM", "SEOBNRv5PHM")
]

DFT_PARAMETERS = [
    {
        "mass_1": 40.0,
        "mass_2": 32.0,
        "a_1": 0.6,
        "a_2": 0.4,
        "tilt_1": 0.9,
        "tilt_2": 1.7,
        "phi_12": 2.1,
        "phi_jl": 0.7,
        "luminosity_distance": 1000.0,
        "theta_jn": 0.9,
        "phase": 1.3,
        "geocent_time": 0.0,
    },
    {
        "mass_1": 60.0,
        "mass_2": 15.0,
        "a_1": 0.2,
        "a_2": 0.8,
        "tilt_1": 2.4,
        "tilt_2": 0.3,
        "phi_12": 5.0,
        "phi_jl": 3.4,
        "luminosity_distance": 1000.0,
        "theta_jn": 2.2,
        "phase": 5.1,
        "geocent_time": 0.0,
    },
]

PHASE_SHIFTS = [0.0, 0.83, 3.7, 5.9]


@pytest.fixture
def dft_wfg_pair(uniform_fd_domain, approximant):
    """Generators differing only in use_dft_phase_decomposition."""
    if approximant == "SEOBNRv5PHM":
        # ell_max comes from the model, which reports max_ell_returned.
        wfg_class, mode_list = NewInterfaceWaveformGenerator, None
    else:
        # XPHM's default mode content. The LAL path has no model to ask, so
        # mode_list is what sizes the phase grid.
        wfg_class, mode_list = WaveformGenerator, [
            (2, 2),
            (2, 1),
            (3, 3),
            (3, 2),
            (4, 4),
        ]
    kwargs = dict(
        approximant=approximant,
        domain=uniform_fd_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
        mode_list=mode_list,
    )
    return (
        wfg_class(**kwargs, use_dft_phase_decomposition=True),
        wfg_class(**kwargs, use_dft_phase_decomposition=False),
    )


@pytest.fixture
def dft_vs_standard_tolerance(approximant):
    """Largest acceptable mismatch between the DFT and individual-mode routes.

    This bound is loose, but it does not mean the DFT route is the less accurate
    of the two. For XPHM it is the other way round. The `tolerances` fixture above
    already documents why the individual-mode route disagrees with a direct
    polarization call -- it needs the magnitude of the orbital angular momentum,
    which dingo computes to a different order than IMRPhenomXPHM does -- and
    accepts up to 2e-2 for it. The DFT route does not inherit that: it is
    assembled from the model's own polarization routine (Appendix C of
    arXiv:2004.06503) and so reproduces it to round-off. Measured against a direct
    call, the DFT route gives a mismatch of 3e-16 and an amplitude agreeing to
    1e-15, where the individual-mode route gives 5e-8 and 1.1e-4. So the
    disagreement bounded here is the individual-mode route's, already known and
    already tolerated above.

    For SEOBNRv5PHM both routes track the model closely and the difference is a
    ~1 ns offset from epoch rounding plus ~6e-6 rad of phase scatter, from
    conditioning and FFT-ing the polarizations rather than the individual modes.

    Either way the bound is set by the weakest m-components: the absolute error is
    roughly common across m while the amplitudes span five orders of magnitude, so
    the relative error is largest exactly where the component contributes least.
    Measured maxima are 1.8e-5 (XPHM) and 6.9e-7 (SEOBNRv5PHM).
    """
    return 1e-4 if approximant == "IMRPhenomXPHM" else 1e-5


@pytest.mark.parametrize("approximant", dft_approximant_list)
def test_dft_reconstructs_phase_shift(dft_wfg_pair, uniform_fd_domain):
    """The DFT m-components reproduce a phase-shifted waveform.

    This is the invariant the DFT inversion has to satisfy, and it pins down the
    sign and ordering of the exp(-i * m * phi_c) factors; getting either wrong
    would otherwise only show up during synthetic-phase inference. The phase
    shifts deliberately fall between grid points, so passing requires the whole
    trigonometric polynomial to be right, not just the sampled values.

    get_mismatch normalises, so it cannot see an overall scale error. The
    amplitude is therefore checked separately: measured departures from unity are
    1e-15 (XPHM) and 3e-9 (SEOBNRv5PHM).
    """
    wfg_dft, _ = dft_wfg_pair
    min_idx = uniform_fd_domain.min_idx

    for p in DFT_PARAMETERS:
        pol_m = wfg_dft.generate_hplus_hcross_m(p)
        for phase_shift in PHASE_SHIFTS:
            pol = sum_contributions_m(pol_m, phase_shift=phase_shift)
            pol_ref = wfg_dft.generate_hplus_hcross(
                {**p, "phase": p["phase"] + phase_shift}
            )
            for name in pol:
                mismatch = get_mismatch(
                    pol[name],
                    pol_ref[name],
                    uniform_fd_domain,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                assert mismatch < 1e-9, f"{name}, phase_shift={phase_shift}"

                amplitude_ratio = np.linalg.norm(pol[name][min_idx:]) / np.linalg.norm(
                    pol_ref[name][min_idx:]
                )
                assert (
                    abs(amplitude_ratio - 1) < 1e-6
                ), f"{name}, phase_shift={phase_shift}, ratio={amplitude_ratio}"


@pytest.mark.parametrize("approximant", dft_approximant_list)
def test_dft_matches_standard_path(
    dft_wfg_pair, uniform_fd_domain, dft_vs_standard_tolerance
):
    """The DFT route and the individual-mode route return the same m-components."""
    wfg_dft, wfg_std = dft_wfg_pair

    for p in DFT_PARAMETERS:
        pol_m_dft = wfg_dft.generate_hplus_hcross_m(p)
        pol_m_std = wfg_std.generate_hplus_hcross_m(p)

        assert set(pol_m_dft) == set(pol_m_std)
        for m, pol_std in pol_m_std.items():
            for name in pol_std:
                mismatch = get_mismatch(
                    pol_m_dft[m][name],
                    pol_std[name],
                    uniform_fd_domain,
                    asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
                )
                assert mismatch < dft_vs_standard_tolerance, f"m={m}, {name}"


@pytest.mark.parametrize("approximant", dft_approximant_list)
def test_dft_phase_grid_ignores_transform(dft_wfg_pair):
    """self.transform must stay out of the DFT phase grid.

    generate_hplus_hcross_m() never applies self.transform, but the DFT route
    builds its grid from generate_hplus_hcross(), which does unless post-processing
    is disabled. Applying it per grid point would corrupt the m-components.
    """
    wfg_dft, _ = dft_wfg_pair
    pol_m = wfg_dft.generate_hplus_hcross_m(DFT_PARAMETERS[0])

    wfg_dft.transform = lambda wf_dict: {k: 2.0 * v for k, v in wf_dict.items()}
    pol_m_with_transform = wfg_dft.generate_hplus_hcross_m(DFT_PARAMETERS[0])

    for m, pol in pol_m.items():
        for name, expected in pol.items():
            np.testing.assert_array_equal(pol_m_with_transform[m][name], expected)


def test_default_ell_max_matches_mode_content(uniform_fd_domain):
    """DEFAULT_ELL_MAX must match what the approximant actually returns.

    Users normally do not pass mode_list -- the settings stored with a trained
    network carry none -- so the phase grid is sized from this table. Too small an
    entry would alias the m-components together silently, so pin it against the
    modes the approximant really produces. SEOBNRv5PHM is covered at runtime
    instead, by the max_ell_returned check in _generate_multi_phase_fd_pols.
    """
    wfg = WaveformGenerator(
        approximant="IMRPhenomXPHM",
        domain=uniform_fd_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )
    hlm_fd, _ = wfg.generate_FD_modes_LO(DFT_PARAMETERS[0])
    assert max(ell for ell, _ in hlm_fd) == DEFAULT_ELL_MAX["IMRPhenomXPHM"]


def test_unknown_approximant_is_an_error_not_a_guess(uniform_fd_domain):
    """An approximant with no tabulated default and no mode_list must fail loudly
    rather than fall back to a grid that may be too short."""
    wfg = WaveformGenerator(
        approximant="IMRPhenomXAS",
        domain=uniform_fd_domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
        use_dft_phase_decomposition=True,
    )
    with pytest.raises(ValueError, match="No default ell_max"):
        wfg._get_ell_max()
