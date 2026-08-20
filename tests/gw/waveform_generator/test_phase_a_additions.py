"""
Tests for Phase A additions to the new-style WaveformGenerator API:

- catch_waveform_errors kwarg on generate_hplus_hcross
- extra_kwargs passthrough to gwsignal
- self.transform slot applied in _apply_post_generation
- base_domain property on NewWaveformGenerator
"""

import numpy as np
import pytest

from dingo.gw.domains import (
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
)
from dingo.gw.waveform_generator import polarization_functions
from dingo.gw.waveform_generator.new_api import (
    build_waveform_generator,
)
from dingo.gw.waveform_generator.polarizations import Polarization
from dingo.gw.waveform_generator.waveform_parameters import RandomWaveformParameters


@pytest.fixture
def ufd() -> UniformFrequencyDomain:
    return UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)


@pytest.fixture
def mfd(ufd) -> MultibandedFrequencyDomain:
    return MultibandedFrequencyDomain(
        nodes=[20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1024.0],
        delta_f_initial=0.125,
        base_domain=ufd,
    )


@pytest.fixture
def wfg(ufd):
    return build_waveform_generator(
        {"approximant": "RandomApproximant", "f_ref": 20.0}, ufd
    )


@pytest.fixture
def params() -> RandomWaveformParameters:
    return RandomWaveformParameters(
        mass_1=36.0, mass_2=29.0, luminosity_distance=1000.0, phase=0.5
    )


class TestCatchWaveformErrors:
    """Verify that catch_waveform_errors traps LAL 'Input domain error' -> NaN pol."""

    _EDOM_MSG = "Internal function call failed: Input domain error"

    def _raise_edom(self, *_args, **_kwargs):
        raise RuntimeError(self._EDOM_MSG)

    def _raise_other(self, *_args, **_kwargs):
        raise RuntimeError("some unrelated failure")

    def test_default_raises(self, wfg, params, monkeypatch):
        monkeypatch.setattr(
            polarization_functions, "random_inspiral_FD", self._raise_edom
        )
        with pytest.raises(RuntimeError, match="Input domain error"):
            wfg.generate_hplus_hcross(params)

    def test_catch_returns_nan_polarization(self, wfg, ufd, params, monkeypatch):
        monkeypatch.setattr(
            polarization_functions, "random_inspiral_FD", self._raise_edom
        )
        with pytest.warns(UserWarning, match="Evaluating the waveform failed"):
            pol = wfg.generate_hplus_hcross(params, catch_waveform_errors=True)
        assert isinstance(pol, Polarization)
        assert pol.h_plus.shape == (len(ufd),)
        assert pol.h_cross.shape == (len(ufd),)
        assert np.all(np.isnan(pol.h_plus))
        assert np.all(np.isnan(pol.h_cross))

    def test_catch_reraises_non_edom(self, wfg, params, monkeypatch):
        monkeypatch.setattr(
            polarization_functions, "random_inspiral_FD", self._raise_other
        )
        with pytest.raises(RuntimeError, match="unrelated"):
            wfg.generate_hplus_hcross(params, catch_waveform_errors=True)


class TestExtraKwargs:
    """Verify that extra_kwargs propagates from build_waveform_generator to the
    WaveformGeneratorParameters and is available for gwsignal-backed subclasses."""

    def test_default_empty(self, wfg):
        assert wfg._waveform_gen_params.extra_kwargs == {}

    def test_passed_through_build(self, ufd):
        w = build_waveform_generator(
            {
                "approximant": "RandomApproximant",
                "f_ref": 20.0,
                "extra_kwargs": {"lmax_nyquist": 4, "postadiabatic": True},
            },
            ufd,
        )
        assert w._waveform_gen_params.extra_kwargs == {
            "lmax_nyquist": 4,
            "postadiabatic": True,
        }

    def test_passed_through_constructor(self, ufd):
        from dingo.gw.approximant import Approximant
        from dingo.gw.waveform_generator.new_api import RandomWaveformGenerator

        w = RandomWaveformGenerator(
            Approximant("RandomApproximant"),
            ufd,
            f_ref=20.0,
            extra_kwargs={"enable_antisymmetric_modes": True},
        )
        assert w._waveform_gen_params.extra_kwargs == {
            "enable_antisymmetric_modes": True
        }


class TestBatchTransformSlot:
    """Verify that self.transform is applied per-waveform in _apply_post_generation."""

    def test_default_none(self, wfg):
        assert wfg.transform is None

    def test_slot_applied(self, wfg, params):
        pol_before = wfg.generate_hplus_hcross(params)

        def double(pol: Polarization) -> Polarization:
            return Polarization(h_plus=pol.h_plus * 2.0, h_cross=pol.h_cross * 2.0)

        wfg.transform = double
        pol_after = wfg.generate_hplus_hcross(params)

        assert np.allclose(pol_after.h_plus, pol_before.h_plus * 2.0)
        assert np.allclose(pol_after.h_cross, pol_before.h_cross * 2.0)


class TestBaseDomain:
    """Verify base_domain property returns the right underlying domain."""

    def test_ufd_returns_self(self, wfg, ufd):
        assert wfg.base_domain is ufd

    def test_mfd_returns_base(self, mfd, ufd):
        w = build_waveform_generator(
            {"approximant": "RandomApproximant", "f_ref": 20.0}, mfd
        )
        assert w.base_domain is ufd

    def test_domain_property(self, wfg, ufd):
        assert wfg.domain is ufd
