"""
Tests for new-style build_waveform_generator factory.

Ported from dingo-waveform tests/test_build_waveform_generator.py.
"""

import json

import pytest
import yaml

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator.new_api import (
    NewWaveformGenerator,
    build_waveform_generator,
)
from dingo.gw.waveform_generator.waveform_generator_parameters import (
    WaveformGeneratorParameters,
)


@pytest.fixture
def domain():
    """Create a test frequency domain."""
    return UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)


@pytest.fixture
def config_dict():
    """Create a test configuration dictionary."""
    return {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
        },
        "waveform_generator": {
            "approximant": "IMRPhenomPv2",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        },
    }


@pytest.fixture
def config_file_json(config_dict, tmp_path):
    """Create a temporary JSON config file."""
    file_path = tmp_path / "config.json"
    with open(file_path, "w") as f:
        json.dump(config_dict, f)
    return file_path


@pytest.fixture
def config_file_yaml(config_dict, tmp_path):
    """Create a temporary YAML config file."""
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_dict, f)
    return file_path


def _assert_waveform_generator(generator):
    """Helper function to assert common properties of a waveform generator."""
    assert isinstance(generator, NewWaveformGenerator)
    assert generator._waveform_gen_params.approximant == "IMRPhenomPv2"
    assert generator._waveform_gen_params.f_ref == 20.0
    assert generator._waveform_gen_params.spin_conversion_phase == 0.0
    assert isinstance(generator._waveform_gen_params.domain, UniformFrequencyDomain)
    assert generator._waveform_gen_params.domain.f_min == 20.0
    assert generator._waveform_gen_params.domain.f_max == 1024.0
    assert generator._waveform_gen_params.domain.delta_f == 0.125


def test_build_waveform_generator_from_dict(domain):
    """Test building a waveform generator from a dictionary."""
    params_dict = {
        "approximant": "IMRPhenomPv2",
        "f_ref": 20.0,
        "spin_conversion_phase": 0.0,
    }
    generator = build_waveform_generator(params_dict, domain)
    _assert_waveform_generator(generator)


def test_build_waveform_generator_from_json_file(config_file_json):
    """Test building a waveform generator from a JSON file."""
    generator = build_waveform_generator(config_file_json)
    _assert_waveform_generator(generator)


def test_build_waveform_generator_from_yaml_file(config_file_yaml):
    """Test building a waveform generator from a YAML file."""
    generator = build_waveform_generator(config_file_yaml)
    _assert_waveform_generator(generator)


def test_build_waveform_generator_invalid_file_format(tmp_path):
    """Test that building from an invalid file format raises an error."""
    file_path = tmp_path / "config.txt"
    file_path.write_text("invalid format")
    with pytest.raises(ValueError, match="Unsupported file format"):
        build_waveform_generator(file_path)


def test_build_waveform_generator_missing_keys(tmp_path):
    """Test that building from a file with missing keys raises an error."""
    file_path = tmp_path / "config.json"
    with open(file_path, "w") as f:
        json.dump({"domain": {}}, f)
    with pytest.raises(ValueError):
        build_waveform_generator(file_path)


def test_build_waveform_generator_invalid_dict(domain):
    """Test that building from an invalid dictionary raises an error."""
    with pytest.raises(ValueError):
        build_waveform_generator({"invalid": "params"}, domain)
