"""
Port of tests/gw/waveform_generator/test_wfg.py::test_new_interface_extra_kwargs
to the new WaveformGenerator API.

Verifies that `extra_kwargs` supplied at construction time are copied through
to the params dict passed to gwsignal's GenerateFDWaveform/GenerateTDWaveform.
"""

import pytest

pytest.importorskip("pyseobnr")

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator import polarization_functions
from dingo.gw.waveform_generator.api import build_waveform_generator
from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters


class _WaveformSpy:
    """Intercepts gwsignal's GenerateFDWaveform to capture the params dict."""

    def __init__(self):
        self.captured_params = None

    def __call__(self, params, generator):
        self.captured_params = params
        # Return a minimal duck-typed object to short-circuit the caller.
        raise _SpyDone(params)


class _SpyDone(Exception):
    def __init__(self, params):
        super().__init__("captured")
        self.params = params


def test_extra_kwargs_forwarded_to_gwsignal(monkeypatch):
    domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.25)
    wfg = build_waveform_generator(
        {
            "approximant": "SEOBNRv5PHM",
            "f_ref": 20.0,
            "extra_kwargs": {
                "lmax_nyquist": 3,
                "postadiabatic": True,
                "postadiabatic_type": "analytic",
                "enable_antisymmetric_modes": True,
                "antisymmetric_modes_hm": True,
            },
        },
        domain,
    )

    spy = _WaveformSpy()
    monkeypatch.setattr(
        polarization_functions.gwsignal_generateFDWaveform.waveform,
        "GenerateFDWaveform",
        spy,
    )

    params = BBHWaveformParameters(
        mass_1=30.0,
        mass_2=25.0,
        luminosity_distance=100.0,
        theta_jn=1.0,
        phase=0.5,
        a_1=0.3,
        a_2=0.2,
        tilt_1=0.5,
        tilt_2=0.3,
        phi_12=1.0,
        phi_jl=0.3,
        geocent_time=0.0,
    )

    with pytest.raises(_SpyDone) as exc_info:
        wfg.generate_hplus_hcross(params)

    captured = exc_info.value.params
    assert captured["lmax_nyquist"] == 3
    assert captured["postadiabatic"] is True
    assert captured["postadiabatic_type"] == "analytic"
    assert captured["enable_antisymmetric_modes"] is True
    assert captured["antisymmetric_modes_hm"] is True
