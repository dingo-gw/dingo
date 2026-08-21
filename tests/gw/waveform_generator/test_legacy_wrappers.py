"""
Regression tests for the deprecated WFG legacy wrappers.

Every wrapper must:
1. emit a DeprecationWarning when used,
2. produce output that is equivalent to a natural-API call (modulo the
   legacy dict shape).
"""

import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator import (
    build_waveform_generator,
    sum_contributions_m as natural_sum_contributions_m,
)
from dingo.gw.waveform_generator.legacy import (
    NewInterfaceWaveformGenerator,
    WaveformGenerator as LegacyWaveformGenerator,
    generate_waveforms_parallel as legacy_generate_waveforms_parallel,
    sum_contributions_m as legacy_sum_contributions_m,
)
from dingo.gw.waveform_generator.polarizations import Polarization
from dingo.gw.waveform_generator.waveform_parameters import RandomWaveformParameters


@pytest.fixture
def domain():
    return UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.125)


@pytest.fixture
def theta():
    return {
        "mass_1": 36.0,
        "mass_2": 29.0,
        "luminosity_distance": 1000.0,
        "phase": 0.5,
    }


class TestLegacyWaveformGenerator:
    def test_emits_deprecation_warning(self, domain):
        with pytest.warns(DeprecationWarning, match="Dict-in"):
            LegacyWaveformGenerator("RandomApproximant", domain, f_ref=20.0)

    def test_generate_matches_natural(self, domain, theta):
        with pytest.warns(DeprecationWarning):
            legacy = LegacyWaveformGenerator(
                "RandomApproximant", domain, f_ref=20.0
            )
        natural = build_waveform_generator(
            {"approximant": "RandomApproximant", "f_ref": 20.0}, domain
        )

        legacy_out = legacy.generate_hplus_hcross(theta)
        natural_out = natural.generate_hplus_hcross(
            RandomWaveformParameters(**theta)
        )

        assert isinstance(legacy_out, dict)
        assert set(legacy_out.keys()) == {"h_plus", "h_cross"}
        assert np.allclose(legacy_out["h_plus"], natural_out.h_plus)
        assert np.allclose(legacy_out["h_cross"], natural_out.h_cross)

    def test_new_interface_alias_warns(self, domain):
        with pytest.warns(DeprecationWarning, match="gwsignal-dispatch"):
            NewInterfaceWaveformGenerator("RandomApproximant", domain, f_ref=20.0)


class TestLegacySumContributionsM:
    def test_emits_deprecation_warning(self):
        x_m = {
            2: {"h_plus": np.ones(4, dtype=complex), "h_cross": np.zeros(4, dtype=complex)},
            -2: {"h_plus": np.ones(4, dtype=complex), "h_cross": np.zeros(4, dtype=complex)},
        }
        with pytest.warns(DeprecationWarning, match="dict-of-dicts"):
            legacy_sum_contributions_m(x_m, phase_shift=0.0)

    def test_result_matches_polarization_version(self):
        # Build parallel per-polarization dict and Polarization dict.
        arr_plus = np.array([1.0, 2.0, 3.0], dtype=complex)
        arr_cross = np.array([0.1, 0.2, 0.3], dtype=complex)
        as_dict = {
            2: {"h_plus": arr_plus, "h_cross": arr_cross},
            -2: {"h_plus": arr_plus.conj(), "h_cross": arr_cross.conj()},
        }
        as_pols = {
            2: Polarization(h_plus=arr_plus, h_cross=arr_cross),
            -2: Polarization(h_plus=arr_plus.conj(), h_cross=arr_cross.conj()),
        }

        with pytest.warns(DeprecationWarning):
            legacy_out = legacy_sum_contributions_m(as_dict, phase_shift=0.3)
        natural_out = natural_sum_contributions_m(as_pols, phase_shift=0.3)

        assert np.allclose(legacy_out["h_plus"], natural_out.h_plus)
        assert np.allclose(legacy_out["h_cross"], natural_out.h_cross)


class TestLegacyGenerateWaveformsParallel:
    def test_emits_deprecation_warning(self, domain):
        with pytest.warns(DeprecationWarning):
            wfg = LegacyWaveformGenerator("RandomApproximant", domain, f_ref=20.0)
        parameters = pd.DataFrame(
            {
                "mass_1": [30.0, 40.0],
                "mass_2": [25.0, 35.0],
                "luminosity_distance": [500.0, 500.0],
                "phase": [0.1, 0.2],
            }
        )
        with pytest.warns(DeprecationWarning, match="legacy signature"):
            legacy_generate_waveforms_parallel(wfg, parameters)

    def test_returns_dict_of_stacked_arrays(self, domain):
        with pytest.warns(DeprecationWarning):
            wfg = LegacyWaveformGenerator("RandomApproximant", domain, f_ref=20.0)
        parameters = pd.DataFrame(
            {
                "mass_1": [30.0, 40.0],
                "mass_2": [25.0, 35.0],
                "luminosity_distance": [500.0, 500.0],
                "phase": [0.1, 0.2],
            }
        )
        with pytest.warns(DeprecationWarning):
            out = legacy_generate_waveforms_parallel(wfg, parameters)
        assert set(out.keys()) == {"h_plus", "h_cross"}
        assert out["h_plus"].shape == (2, len(domain))
        assert out["h_cross"].shape == (2, len(domain))
