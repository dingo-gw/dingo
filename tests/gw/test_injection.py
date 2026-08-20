import pytest
import numpy as np

from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.injection import GWSignal


def _sum_over_m(per_m: dict, phase_shift: float) -> dict:
    """Sum contributions over m for a per-detector nested dict.

    Input shape: {m: {"H1": arr, "L1": arr, ...}}. Returns
    {"H1": sum_m arr * exp(-1j m phi), "L1": ...}. The new-API
    sum_contributions_m in polarizations.py is Polarization-typed; the
    signal_m output here is per-detector, so we inline the sum locally.
    """
    keys = next(iter(per_m.values())).keys()
    result = {k: 0.0 for k in keys}
    for k in keys:
        for m, sample in per_m.items():
            result[k] = result[k] + sample[k] * np.exp(-1j * m * phase_shift)
    return result


@pytest.fixture
def signal_setup_EOB():
    wfg_kwargs = {
        "approximant": "SEOBNRv4PHM",
        "f_ref": 10.0,
        "f_start": 10.0,
        "spin_conversion_phase": 0,
    }
    domain = build_domain(
        {
            "type": "UniformFrequencyDomain",
            "f_min": 20,
            "f_max": 1024,
            "delta_f": 0.125,
        }
    )
    ifo_list = ["H1", "L1"]
    t_ref = 1126259462.4
    signal = GWSignal(wfg_kwargs, domain, domain, ifo_list, t_ref)
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


def test_signal_m_EOB(signal_setup_EOB, BBH_parameters):
    signal = signal_setup_EOB
    p = BBH_parameters
    domain = signal.waveform_generator.domain

    phase_shift = np.random.uniform(high=2 * np.pi)
    waveform_ref = signal.signal({**p, "phase": p["phase"] + phase_shift})["waveform"]
    waveform_m = signal.signal_m(p)
    waveform_m = {k: v["waveform"] for k, v in waveform_m.items()}
    waveform = _sum_over_m(waveform_m, phase_shift=phase_shift)
    mismatches = [
        get_mismatch(
            waveform_ref[k],
            waveform[k],
            domain,
            asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
        )
        for k in waveform.keys()
    ]
    assert np.max(mismatches) < 1e-4
