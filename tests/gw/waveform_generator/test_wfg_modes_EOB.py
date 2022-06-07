"""
This tests that the waveform generator provides the correct waveform for EOB if the
polarizations are returned with respect to the individual modes. This is important when
treating the phase parameter as an extrinsic parameter.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from dingo.gw.domains import build_domain

from dingo.gw.waveform_generator import WaveformGenerator, sum_fd_mode_contributions
from dingo.gw.gwutils import get_mismatch


@pytest.fixture
def uniform_fd_domain():
    domain_settings = {
        "type": "FrequencyDomain",
        "f_min": 20.0,
        "f_max": 2048.0,
        "delta_f": 0.125,
    }
    domain = build_domain(domain_settings)
    return domain


@pytest.fixture
def BBH_parameters():
    parameters = {
        "mass_1": 60.29442201204798,
        "mass_2": 25.460299253933126,
        "phase": 2.346269257440926,
        "a_1": 0.07104636316747037,
        "a_2": 0.7853578509086726,
        "tilt_1": 1.8173336549500292,
        "tilt_2": 0.4380213394743055,
        "phi_12": 5.892609139936818,
        "phi_jl": 1.6975651971466297,
        "theta_jn": 1.0724395559873239,
        "luminosity_distance": 100.0,
        "geocent_time": 0.0,
    }
    return parameters


def test_mode_recombination(uniform_fd_domain, BBH_parameters):
    """
    Basic test that calling wfg.generate_h_plus_h_cross(parameters) returns the same
    result as calling wfg.generate_h_plus_h_cross_modes(parameters) and then summing over
    the modes. This tests the mode generation, recombination with spherical harmonics,
    and the FFT (including tapering) end-to-end.
    """
    visualize = False

    domain = uniform_fd_domain
    parameters = BBH_parameters
    waveform_generator = WaveformGenerator(
        "SEOBNRv4PHM",
        domain,
        20.0,
        f_start=10.0,
    )
    pol_dict_ref = waveform_generator.generate_hplus_hcross(parameters)
    pol_dict_modes = waveform_generator.generate_hplus_hcross_modes(parameters)
    pol_dict_summed = sum_fd_mode_contributions(pol_dict_modes, delta_phi=0.0)

    if visualize:
        plt.plot(pol_dict_ref["h_plus"])
        plt.plot(pol_dict_summed["h_plus"])
        plt.plot(pol_dict_summed["h_plus"] - pol_dict_ref["h_plus"])
        plt.show()

    for pol in ["h_plus", "h_cross"]:
        assert get_mismatch(pol_dict_ref[pol], pol_dict_summed[pol], domain) < 1e-4


def test_mode_recombination_with_phase(uniform_fd_domain, BBH_parameters):
    """
    Test that calling wfg.generate_h_plus_h_cross(parameters) returns the same
    result as calling wfg.generate_h_plus_h_cross_modes({**parameters, phase=0}) and
    then summing over the modes with
    sum_polarization_modes(pol_dict_modes, delta_phi=parameters["phase"]).

    Note: for identical results, one needs to set DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS =
    True, since otherwise phase enters not just via spherical harmonics, but also in
    the calculation for the cartesian spins (it rotates the spins in the x-y plan,
    so while sx**2 + sy**2 = const., but sx and sy individually depend on phase). While
    the latter effect in principle needs to be accounted for, it is useful to neglect
    it here to test that the phase transformation of the spherical harmonics is
    implemented properly in sum_polarization_modes.
    """
    visualize = False

    domain = uniform_fd_domain
    parameters = BBH_parameters

    DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS=True
    # in this case, the phase only comes in via the spherical harmonics,
    # so the results of
    #   (a) wfg.generate_hplus_hcross(parameters)
    #   (b) sum_polarization_modes(
    #           wfg.generate_hplus_hcross_modes({**parameters, "phase": 0.0}),
    #           delta_phi=parameters["phase"],
    #       )
    # should match perfectly.
    waveform_generator = WaveformGenerator(
        "SEOBNRv4PHM",
        domain,
        20.0,
        f_start=10.0,
        DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS=DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS,
    )
    pol_dict_ref = waveform_generator.generate_hplus_hcross(parameters)
    pol_dict_modes = waveform_generator.generate_hplus_hcross_modes(
        {**parameters, "phase": 0.0}
    )
    pol_dict_summed = sum_fd_mode_contributions(
        pol_dict_modes, delta_phi=parameters["phase"]
    )

    if visualize:
        x = domain()
        m = get_mismatch(pol_dict_ref['h_plus'], pol_dict_summed['h_plus'], domain)
        plt.title(f"h_plus. Mismatch: {m:.2e}.")
        plt.plot(
            x,
            pol_dict_ref["h_plus"],
            label="Reference (neglecting phase for cartesian spins)"
        )
        plt.plot(x, pol_dict_summed["h_plus"], label="Summed from modes")
        plt.plot(x, pol_dict_summed["h_plus"] - pol_dict_ref["h_plus"], label="diff")
        plt.xlim((0, 100))
        plt.xlabel("f in Hz")
        plt.legend()
        plt.show()

    for pol in ["h_plus", "h_cross"]:
        assert get_mismatch(pol_dict_ref[pol], pol_dict_summed[pol], domain) < 1e-4

    # for comparison: naive baseline (i.e., exp(2i*phase) transformation) that
    # assumes that only (2, 2) mode contributes.
    pol_dict_naive = waveform_generator.generate_hplus_hcross(
        {**parameters, "phase": 0.0}
    )
    pol_dict_naive = {
        k: v * np.exp(2j * parameters["phase"]) for k, v in pol_dict_naive.items()
    }
    mismatches = []
    mismatches_naive = []
    for pol in ["h_plus", "h_cross"]:
        mismatches.append(get_mismatch(pol_dict_ref[pol], pol_dict_summed[pol], domain))
        mismatches_naive.append(get_mismatch(pol_dict_ref[pol], pol_dict_naive[pol], domain))
    # mismatches should be significantly smaller than mismatches_naive
    assert np.mean(mismatches) < np.mean(mismatches_naive)


    DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS=False
    # in this case, the phase also comes in via the cartesian spins. In
    #   (a) wfg.generate_hplus_hcross(parameters)
    #   (b) sum_polarization_modes(
    #           wfg.generate_hplus_hcross_modes({**parameters, "phase": 0.0}),
    #           delta_phi=parameters["phase"],
    #       )
    # version (b) thus computes the modes with a wrong orientation of the
    # cartesian spins in the xy-planes, so we do expect a somewhat larger
    # deviation here.
    # However, the main effect of the phase should still come in via the
    # spherical harmonics, so if the higher modes are relevant, option (b)
    # should still be much better than the naive baseline with exp(2i*phase)
    # transformation.
    waveform_generator = WaveformGenerator(
        "SEOBNRv4PHM",
        domain,
        20.0,
        f_start=10.0,
        DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS=DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS,
    )
    pol_dict_ref = waveform_generator.generate_hplus_hcross(parameters)
    pol_dict_modes = waveform_generator.generate_hplus_hcross_modes(
        {**parameters, "phase": 0.0}
    )
    pol_dict_summed = sum_fd_mode_contributions(
        pol_dict_modes, delta_phi=parameters["phase"]
    )

    if visualize:
        x = domain()
        m = get_mismatch(pol_dict_ref['h_plus'], pol_dict_summed['h_plus'], domain)
        plt.title(f"h_plus. Mismatch: {m:.2e}.")
        plt.plot(
            x,
            pol_dict_ref["h_plus"],
            label="Reference (using phase for cartesian spins)"
        )
        plt.plot(x, pol_dict_summed["h_plus"], label="Summed from modes")
        plt.plot(x, pol_dict_summed["h_plus"] - pol_dict_ref["h_plus"], label="diff")
        plt.xlim((0, 100))
        plt.xlabel("f in Hz")
        plt.legend()
        plt.show()

    # for comparison: naive baseline (i.e., exp(2i*phase) transformation) that
    # assumes that only (2, 2) mode contributes.
    pol_dict_naive = waveform_generator.generate_hplus_hcross(
        {**parameters, "phase": 0.0}
    )
    pol_dict_naive = {
        k: v * np.exp(2j * parameters["phase"]) for k, v in pol_dict_naive.items()
    }

    if visualize:
        x = domain()
        m = get_mismatch(pol_dict_ref['h_plus'], pol_dict_naive['h_plus'], domain)
        plt.title(f"h_plus. Mismatch: {m:.2e}.")
        plt.plot(
            x,
            pol_dict_ref["h_plus"],
            label="Reference (using phase for cartesian spins)"
        )
        plt.plot(x, pol_dict_naive["h_plus"], label="Naive exp(2i*phase) transformation")
        plt.plot(x, pol_dict_naive["h_plus"] - pol_dict_ref["h_plus"], label="diff")
        plt.xlim((0, 100))
        plt.xlabel("f in Hz")
        plt.legend()
        plt.show()

    mismatches = []
    mismatches_naive = []
    for pol in ["h_plus", "h_cross"]:
        mismatches.append(
            get_mismatch(pol_dict_ref[pol], pol_dict_summed[pol], domain))
        mismatches_naive.append(
            get_mismatch(pol_dict_ref[pol], pol_dict_naive[pol], domain))
    # mismatches should be significantly smaller than mismatches_naive
    assert np.mean(mismatches) < np.mean(mismatches_naive)


