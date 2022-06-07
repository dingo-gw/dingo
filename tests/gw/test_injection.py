import pytest
import numpy as np
import time
import matplotlib.pyplot as plt

from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.inference.injection import GWSignal
from dingo.gw.waveform_generator import sum_fd_mode_contributions


@pytest.fixture
def signal_setup_EOB():
    wfg_kwargs = {
        "approximant": "SEOBNRv4PHM",
        "f_ref": 20.0,
        "f_start": 10.0,
        "DEBUG_FIX_PHASE_FOR_CARTESIAN_SPINS": False,
    }
    wfg_domain = build_domain(
        {
            "type": "FrequencyDomain",
            "f_min": 20,
            "f_max": 1024,
            "delta_f": 0.125,
            "window_factor": 1.0,
        }
    )
    data_domain = build_domain(
        {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
            "window_factor": 0.9374713897717841,
        }
    )
    ifo_list = ["H1", "L1"]
    t_ref = 1126259462.4
    signal = GWSignal(wfg_kwargs, wfg_domain, data_domain, ifo_list, t_ref)
    return signal


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
        "ra": 1.0,
        "dec": 2.0,
        "psi": 2.5,
    }
    return parameters


def test_signal_from_cached_polarizations(signal_setup_EOB, BBH_parameters):
    """
    Test GWSignal.signal_from_cached_polarizations method.
    """
    visualize = False

    signal = signal_setup_EOB
    theta = BBH_parameters
    # compute signals from cached polarizations
    t0 = time.time()
    signal_0, pol_0 = signal.signal_from_cached_polarizations(
        {**theta, "phase": 0},
    )
    delta_t0 = time.time() - t0
    t0 = time.time()
    signal_1, pol_1 = signal.signal_from_cached_polarizations(
        {**theta, "phase": 1},
        polarizations_cached=pol_0,
    )
    delta_t1 = time.time() - t0
    # Check that compunting cached signal is significantly faster
    assert delta_t0 > 10 * delta_t1, "Caching does not seem to have an effect."

    # compute reference signals
    signal_0_ref = signal.signal({**theta, "phase": 0})
    signal_1_ref = signal.signal({**theta, "phase": 1})
    mismatch_0 = get_mismatch(
        signal_0["waveform"]["H1"],
        signal_0_ref["waveform"]["H1"],
        signal.waveform_generator.domain,
    )
    mismatch_1 = get_mismatch(
        signal_1["waveform"]["H1"],
        signal_1_ref["waveform"]["H1"],
        signal.waveform_generator.domain,
    )

    assert mismatch_0 < 1e-4
    # We don't expect a very small mismatch_1 here, since the effect of the phase on the
    # cartesian spins is not accounted for.
    assert mismatch_1 < 5e-2

    if visualize:
        x = signal.waveform_generator.domain()
        plt.title(f"mismatch {mismatch_0:.2e}")
        plt.xlim((20, 1024))
        plt.xscale("log")
        plt.plot(x, signal_0["waveform"]["H1"])
        plt.plot(x, signal_0_ref["waveform"]["H1"])
        plt.show()
        plt.title(f"mismatch {mismatch_1:.2e}")
        plt.xlim((20, 1024))
        plt.xscale("log")
        plt.plot(x, signal_1["waveform"]["H1"])
        plt.plot(x, signal_1_ref["waveform"]["H1"])
        plt.show()


def test_signal_modes(signal_setup_EOB, BBH_parameters):
    signal = signal_setup_EOB
    theta = BBH_parameters
    domain = signal.waveform_generator.domain

    waveform_0 = signal.signal(theta)["waveform"]
    waveform_1 = signal.signal_modes(theta)
    waveform_1 = {k: v["waveform"] for k, v in waveform_1.items()}
    waveform_summed_1 = sum_fd_mode_contributions(waveform_1)
    mismatches = [
        get_mismatch(waveform_0[k], waveform_summed_1[k], domain)
        for k in waveform_0.keys()
    ]
    assert np.max(mismatches) < 1e-4
