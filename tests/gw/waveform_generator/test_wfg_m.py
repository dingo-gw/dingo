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


@pytest.fixture
def intrinsic_prior():
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
    prior = build_prior_with_defaults(intrinsic_dict)
    return prior


def test_generate_hplus_hcross_m_SEOBNRv4PHM(uniform_fd_domain, intrinsic_prior):
    domain = uniform_fd_domain
    prior = intrinsic_prior
    wfg = WaveformGenerator(
        "SEOBNRv4PHM",
        domain,
        10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )

    p = prior.sample()
    phase_shift = np.random.uniform(high=2 * np.pi)
    pol_m = wfg.generate_hplus_hcross_m(p)
    pol = sum_contributions_m(pol_m, phase_shift=phase_shift)
    pol_ref = wfg.generate_hplus_hcross({**p, "phase": p["phase"] + phase_shift})

    mismatches = [
        get_mismatch(
            pol[pol_name],
            pol_ref[pol_name],
            wfg.domain,
            asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
        )
        for pol_name in pol
    ]

    # The mismatches are typically be of order 1e-5. This is exclusively due to
    # different tapering. The reference polarizations are tapered and FFTed on the
    # level of polarizations, while for generate_hplus_hcross_m, the tapering and FFT
    # happens on the level of complex modes.
    # We tested the mismatches for 20k waveforms, and the largest mismatch encountered
    # was 7e-4, while almost all mismatches were of order 1e-5.
    assert max(mismatches) < 5e-4


def test_generate_hplus_hcross_m_IMRPhenomXPHM(uniform_fd_domain, intrinsic_prior):
    domain = uniform_fd_domain
    prior = intrinsic_prior
    wfg = WaveformGenerator(
        "IMRPhenomXPHM",
        domain,
        20.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )

    mismatches = []
    for idx in range(10):
        p = prior.sample()
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

    # The mismatches are typically be of order 1e-5 to 1e-9. This comes from the
    # calculation of the magnitude of the orbital angular momentum, which we calculate
    # to a different order the IMRPhenomXPHM. It's tricky to get this exactly right,
    # since there are many different methods for this. But the small mismatches we do
    # get should not have a big effect in practice.
    mismatches = np.array(mismatches)
    assert np.max(mismatches) < 1e-1
    assert np.median(mismatches) < 1e-6

def test_generate_hplus_hcross_m_SEOBNRv5PHM(uniform_fd_domain, intrinsic_prior):
    domain = uniform_fd_domain
    prior = intrinsic_prior
    wfg = NewInterfaceWaveformGenerator(
        approximant="SEOBNRv5PHM",
        domain=domain,
        f_ref=10.0,
        f_start=10.0,
        spin_conversion_phase=0.0,
    )

    p = prior.sample()
    phase_shift = np.random.uniform(high=2 * np.pi)
    pol_m = wfg.generate_hplus_hcross_m(p)
    pol = sum_contributions_m(pol_m, phase_shift=phase_shift)
    pol_ref = wfg.generate_hplus_hcross({**p, "phase": p["phase"] + phase_shift})

    mismatches = [
        get_mismatch(
            pol[pol_name],
            pol_ref[pol_name],
            wfg.domain,
            asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
        )
        for pol_name in pol
    ]

    debug = False
    if debug:
        maxval = max(mismatches)
        idx = mismatches.index(maxval)
        p = list(pol.keys())[idx]
        plt.figure(figsize=(10, 7))
        plt.plot(domain.sample_frequencies, pol[p], label="reconstructed")
        plt.plot(domain.sample_frequencies, pol_ref[p], label="ref")
        plt.plot(domain.sample_frequencies, pol_ref[p] - pol[p], label="diff")
        plt.legend()
        plt.xscale("log")
        plt.xlim((5, 128))
        plt.title(f"{p}, mismatch={maxval}")
        plt.show()

    # The mismatches are typically be of order 1e-5. This is exclusively due to
    # different tapering. The reference polarizations are tapered and FFTed on the
    # level of polarizations, while for generate_hplus_hcross_m, the tapering and FFT
    # happens on the level of complex modes.
    # We tested the mismatches for 20k waveforms, and the largest mismatch encountered
    # was 7e-4, while almost all mismatches were of order 1e-5.
    assert max(mismatches) < 5e-4
