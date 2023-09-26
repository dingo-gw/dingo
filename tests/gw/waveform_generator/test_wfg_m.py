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
from dingo.gw.gwutils import get_mismatch
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults


@pytest.fixture
def uniform_fd_domain():
    domain_settings = {
        "type": "FrequencyDomain",
        "f_min": 10.0,
        "f_max": 2048.0,
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
        return 2e-2, 1e-6

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
# @pytest.mark.parametrize("approximant", ["IMRPhenomXPHM"])
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
            plt.plot(wfg.domain.sample_frequencies, pol_ref[p], label="ref")
            plt.plot(wfg.domain.sample_frequencies, pol_ref[p] - pol[p], label="diff")
            plt.legend()
            plt.xscale("log")
            plt.xlim((5, 128))
            plt.title(f"{p}, mismatch={maxval}")
            plt.show()

    mismatches = np.array(mismatches)

    assert np.max(mismatches) < tolerances[0]
    assert np.median(mismatches) < tolerances[1]
