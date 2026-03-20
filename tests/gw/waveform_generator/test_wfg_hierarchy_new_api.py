"""Tests for the new-style WaveformGenerator class hierarchy.

Ported from dingo-waveform tests/test_wfg_hierarchy.py, adapted for the
new API where only RandomApproximant is fully implemented and LAL-based
subclasses are stubs.
"""

import pytest

from dingo.gw.approximant import Approximant
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator.new_api import (
    GWSignalWaveformGenerator,
    IMRPhenomXPHMWaveformGenerator,
    LALSimWaveformGenerator,
    NewWaveformGenerator,
    RandomWaveformGenerator,
    SEOBNRv4PHMWaveformGenerator,
    _get_waveform_generator_class,
    build_waveform_generator,
)


@pytest.fixture
def domain():
    return UniformFrequencyDomain(f_min=20.0, f_max=512.0, delta_f=0.25)


class TestClassMapping:
    """Test that _get_waveform_generator_class returns the right subclass."""

    def test_random_approximant(self):
        cls = _get_waveform_generator_class(Approximant("RandomApproximant"))
        assert cls is RandomWaveformGenerator

    def test_unknown_defaults_to_lalsim(self):
        cls = _get_waveform_generator_class(Approximant("IMRPhenomD"))
        assert cls is LALSimWaveformGenerator

    def test_imrphenomxphm_maps_to_subclass(self):
        cls = _get_waveform_generator_class(Approximant("IMRPhenomXPHM"))
        assert cls is IMRPhenomXPHMWaveformGenerator
        assert issubclass(cls, LALSimWaveformGenerator)

    def test_seobnrv4phm_maps_to_subclass(self):
        cls = _get_waveform_generator_class(Approximant("SEOBNRv4PHM"))
        assert cls is SEOBNRv4PHMWaveformGenerator
        assert issubclass(cls, LALSimWaveformGenerator)

    def test_seobnrv5phm_maps_to_gwsignal(self):
        cls = _get_waveform_generator_class(Approximant("SEOBNRv5PHM"))
        assert cls is GWSignalWaveformGenerator

    def test_seobnrv5hm_maps_to_gwsignal(self):
        cls = _get_waveform_generator_class(Approximant("SEOBNRv5HM"))
        assert cls is GWSignalWaveformGenerator


class TestBuildWaveformGenerator:
    """Test that build_waveform_generator returns correct subclass instances."""

    def test_builds_random(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "RandomApproximant", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, RandomWaveformGenerator)
        assert isinstance(wfg, NewWaveformGenerator)

    def test_builds_lalsim(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomD", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, LALSimWaveformGenerator)
        assert isinstance(wfg, NewWaveformGenerator)
        assert not isinstance(wfg, RandomWaveformGenerator)

    def test_builds_imrphenomxphm(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomXPHM", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, IMRPhenomXPHMWaveformGenerator)
        assert isinstance(wfg, LALSimWaveformGenerator)
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_builds_seobnrv4phm(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv4PHM", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, SEOBNRv4PHMWaveformGenerator)
        assert isinstance(wfg, LALSimWaveformGenerator)
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_builds_gwsignal(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "SEOBNRv5PHM", "f_ref": 20.0}, domain
        )
        assert isinstance(wfg, GWSignalWaveformGenerator)
        assert isinstance(wfg, NewWaveformGenerator)
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_build_with_optional_params(self, domain):
        wfg = build_waveform_generator(
            {
                "approximant": "RandomApproximant",
                "f_ref": 20.0,
                "spin_conversion_phase": 0.5,
                "f_start": 15.0,
            },
            domain,
        )
        assert isinstance(wfg, RandomWaveformGenerator)
        assert wfg._waveform_gen_params.spin_conversion_phase == 0.5
        assert wfg._waveform_gen_params.f_start == 15.0

    def test_build_missing_approximant(self, domain):
        with pytest.raises(ValueError, match="approximant"):
            build_waveform_generator({"f_ref": 20.0}, domain)

    def test_build_missing_f_ref(self, domain):
        with pytest.raises(ValueError, match="f_ref"):
            build_waveform_generator(
                {"approximant": "RandomApproximant"}, domain
            )

    def test_build_missing_domain(self):
        with pytest.raises(ValueError, match="domain"):
            build_waveform_generator(
                {"approximant": "RandomApproximant", "f_ref": 20.0}
            )


class TestModeSupport:
    """Test mode-separated generation support."""

    def test_random_has_modes(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "RandomApproximant", "f_ref": 20.0}, domain
        )
        assert hasattr(wfg, "generate_hplus_hcross_m")

    def test_plain_lalsim_no_modes(self, domain):
        wfg = build_waveform_generator(
            {"approximant": "IMRPhenomD", "f_ref": 20.0}, domain
        )
        assert not hasattr(wfg, "generate_hplus_hcross_m")


class TestABCNotInstantiable:
    """Test that the ABC cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self, domain):
        with pytest.raises(TypeError):
            NewWaveformGenerator(
                Approximant("RandomApproximant"), domain, 20.0
            )


class TestInheritance:
    """Test the class hierarchy relationships."""

    def test_lalsim_is_wfg(self):
        assert issubclass(LALSimWaveformGenerator, NewWaveformGenerator)

    def test_random_is_wfg(self):
        assert issubclass(RandomWaveformGenerator, NewWaveformGenerator)

    def test_random_is_not_lalsim(self):
        assert not issubclass(RandomWaveformGenerator, LALSimWaveformGenerator)

    def test_seobnrv4phm_is_lalsim(self):
        assert issubclass(SEOBNRv4PHMWaveformGenerator, LALSimWaveformGenerator)

    def test_imrphenomxphm_is_lalsim(self):
        assert issubclass(IMRPhenomXPHMWaveformGenerator, LALSimWaveformGenerator)

    def test_gwsignal_is_wfg(self):
        assert issubclass(GWSignalWaveformGenerator, NewWaveformGenerator)

    def test_gwsignal_is_not_lalsim(self):
        assert not issubclass(GWSignalWaveformGenerator, LALSimWaveformGenerator)
