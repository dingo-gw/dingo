"""
Tests for generate_multibanded_domain.py, evaluate_multibanded_domain.py, and
_multibanded_domain_utils.py.

Tests for the top-level generate_multibanded_domain_settings and
_evaluate_multibanding_main functions (which require waveform generation) are not
included here; those are covered by integration tests.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from bilby.core.prior import DeltaFunction

from dingo.gw.dataset._multibanded_domain_utils import build_extreme_prior
from dingo.gw.dataset.evaluate_multibanded_domain import \
    _evaluate_multibanding_main
from dingo.gw.dataset.generate_multibanded_domain import (
    _build_mfd_for_threshold, _compute_mismatches, _output_settings_path,
    compute_max_decimation_factor,
    compute_waveform_difference_per_decimation_factor, floor_to_power_of_2,
    get_band_nodes_for_adaptive_decimation)
from dingo.gw.domains import MultibandedFrequencyDomain, UniformFrequencyDomain
from dingo.gw.prior import default_intrinsic_dict

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INTRINSIC_PRIOR = default_intrinsic_dict


@pytest.fixture
def settings():
    return {"intrinsic_prior": INTRINSIC_PRIOR.copy()}


@pytest.fixture
def ufd():
    return UniformFrequencyDomain(f_min=20.0, f_max=256.0, delta_f=1.0)


@pytest.fixture
def ufd_2x(ufd):
    return UniformFrequencyDomain(
        f_min=ufd.f_min, f_max=ufd.f_max, delta_f=ufd.delta_f / 2
    )


@pytest.fixture
def mfd(ufd):
    # Two bands: [20, 64) with dec=1, [64, 256) with dec=2
    return MultibandedFrequencyDomain(
        nodes=[20.0, 64.0, 256.0],
        delta_f_initial=1.0,
        base_domain=ufd,
    )


# ---------------------------------------------------------------------------
# floor_to_power_of_2
# ---------------------------------------------------------------------------


class TestFloorToPowerOf2:
    @pytest.mark.parametrize(
        "x, expected",
        [
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 2.0),
            (4.0, 4.0),
            (7.9, 4.0),
            (8.0, 8.0),
            (128.0, 128.0),
            (129.0, 128.0),
        ],
    )
    def test_values(self, x, expected):
        assert floor_to_power_of_2(x) == expected


# ---------------------------------------------------------------------------
# get_band_nodes_for_adaptive_decimation
# ---------------------------------------------------------------------------


class TestGetBandNodesForAdaptiveDecimation:
    def test_all_ones_gives_single_band(self):
        max_dec = np.ones(100)
        initial_ds, nodes = get_band_nodes_for_adaptive_decimation(max_dec)
        assert initial_ds == 1
        assert nodes[0] == 0
        assert nodes[-1] == len(max_dec)

    def test_nodes_start_at_zero(self):
        max_dec = np.concatenate([np.ones(50), 4 * np.ones(50)])
        _, nodes = get_band_nodes_for_adaptive_decimation(max_dec)
        assert nodes[0] == 0

    def test_nodes_monotonically_increasing(self):
        max_dec = np.concatenate([np.ones(32), 2 * np.ones(32), 4 * np.ones(32)])
        _, nodes = get_band_nodes_for_adaptive_decimation(max_dec)
        assert all(nodes[i] < nodes[i + 1] for i in range(len(nodes) - 1))

    def test_global_max_dec_factor_clipping(self):
        max_dec = np.concatenate([np.ones(16), 2 * np.ones(16), 4 * np.ones(32)])
        initial_ds, _ = get_band_nodes_for_adaptive_decimation(
            max_dec, max_dec_factor_global=2
        )
        assert initial_ds <= 2

    def test_raises_for_non_monotonic_input(self):
        max_dec = np.array([1.0, 2.0, 1.0])
        with pytest.raises(ValueError):
            get_band_nodes_for_adaptive_decimation(max_dec)

    def test_raises_for_2d_input(self):
        max_dec = np.ones((10, 10))
        with pytest.raises(ValueError):
            get_band_nodes_for_adaptive_decimation(max_dec)


# ---------------------------------------------------------------------------
# compute_max_decimation_factor
# ---------------------------------------------------------------------------


class TestComputeMaxDecimationFactor:
    def test_shape(self, ufd):
        decimation_factors = np.array([2, 4])
        n = len(ufd()) // 2
        diffs = [np.zeros(n), np.zeros(n)]
        freqs = [np.linspace(0, ufd.f_max, n), np.linspace(0, ufd.f_max, n)]
        result = compute_max_decimation_factor(
            decimation_factors, diffs, freqs, ufd, threshold=0.1
        )
        assert result.shape == (len(ufd()),)

    def test_result_monotonically_nondecreasing(self, ufd):
        # Monotonically decreasing diff → decimation factor increases with frequency
        decimation_factors = np.array([2, 4])
        n = len(ufd()) // 2
        diffs = [np.linspace(1.0, 0.0, n), np.linspace(1.0, 0.0, n)]
        freqs = [np.linspace(0, ufd.f_max, n), np.linspace(0, ufd.f_max, n)]
        result = compute_max_decimation_factor(
            decimation_factors, diffs, freqs, ufd, threshold=0.5
        )
        assert np.all(result[1:] >= result[:-1])

    def test_all_values_among_valid_decimation_factors(self, ufd):
        decimation_factors = np.array([2, 4, 8])
        n = len(ufd()) // 2
        diffs = [np.zeros(n)] * 3
        freqs = [np.linspace(0, ufd.f_max, n)] * 3
        result = compute_max_decimation_factor(
            decimation_factors, diffs, freqs, ufd, threshold=0.1
        )
        valid = np.concatenate([[1], decimation_factors])
        assert all(v in valid for v in np.unique(result))


# ---------------------------------------------------------------------------
# compute_waveform_difference_per_decimation_factor
# ---------------------------------------------------------------------------


class TestComputeWaveformDifferencePerdecimationFactor:
    def test_flat_waveforms_give_zero_difference(self, ufd, ufd_2x):
        n_samples = 5
        waveforms = np.ones((n_samples, len(ufd())))
        waveforms_2x = np.ones((n_samples, len(ufd_2x())))
        decimation_factors = np.array([2, 4])
        diffs, freqs = compute_waveform_difference_per_decimation_factor(
            decimation_factors, waveforms, ufd, waveforms_2x
        )
        for d in diffs:
            assert np.allclose(d, 0.0, atol=1e-10)

    def test_output_length_matches_decimation_factors(self, ufd, ufd_2x):
        n_samples = 5
        waveforms = np.random.randn(n_samples, len(ufd()))
        waveforms_2x = np.random.randn(n_samples, len(ufd_2x()))
        decimation_factors = np.array([2, 4, 8])
        diffs, freqs = compute_waveform_difference_per_decimation_factor(
            decimation_factors, waveforms, ufd, waveforms_2x
        )
        assert len(diffs) == len(decimation_factors)
        assert len(freqs) == len(decimation_factors)

    def test_diffs_are_nonnegative(self, ufd, ufd_2x):
        n_samples = 10
        waveforms = np.random.randn(n_samples, len(ufd()))
        waveforms_2x = np.random.randn(n_samples, len(ufd_2x()))
        decimation_factors = np.array([2])
        diffs, _ = compute_waveform_difference_per_decimation_factor(
            decimation_factors, waveforms, ufd, waveforms_2x
        )
        assert np.all(diffs[0] >= 0)

    def test_diffs_are_monotonically_nonincreasing(self, ufd, ufd_2x):
        # The cumulative max ensures the diff array is non-increasing
        n_samples = 10
        waveforms = np.random.randn(n_samples, len(ufd()))
        waveforms_2x = np.random.randn(n_samples, len(ufd_2x()))
        decimation_factors = np.array([2])
        diffs, _ = compute_waveform_difference_per_decimation_factor(
            decimation_factors, waveforms, ufd, waveforms_2x
        )
        assert np.all(diffs[0][:-1] >= diffs[0][1:])


# ---------------------------------------------------------------------------
# _build_mfd_for_threshold
# ---------------------------------------------------------------------------


class TestBuildMfdForThreshold:
    def test_returns_valid_mfd(self, ufd, ufd_2x):
        n_samples = 5
        waveforms = np.ones((n_samples, len(ufd())))
        waveforms_2x = np.ones((n_samples, len(ufd_2x())))
        decimation_factors = np.array([2, 4])
        diffs, freqs = compute_waveform_difference_per_decimation_factor(
            decimation_factors, waveforms, ufd, waveforms_2x
        )
        mfd = _build_mfd_for_threshold(diffs, freqs, decimation_factors, ufd, 0.01, 2.0)
        assert isinstance(mfd, MultibandedFrequencyDomain)
        assert mfd.num_bands >= 1

    def test_higher_threshold_gives_equal_or_fewer_bins(self, ufd, ufd_2x):
        # Higher threshold → more aggressive decimation → fewer MFD bins
        n_samples = 10
        np.random.seed(0)
        waveforms = np.random.randn(n_samples, len(ufd()))
        waveforms_2x = np.random.randn(n_samples, len(ufd_2x()))
        decimation_factors = np.array([2, 4, 8])
        diffs, freqs = compute_waveform_difference_per_decimation_factor(
            decimation_factors, waveforms, ufd, waveforms_2x
        )
        mfd_conservative = _build_mfd_for_threshold(
            diffs, freqs, decimation_factors, ufd, 1e-4, 2.0
        )
        mfd_aggressive = _build_mfd_for_threshold(
            diffs, freqs, decimation_factors, ufd, 1.0, 2.0
        )
        assert len(mfd_aggressive()) <= len(mfd_conservative())


# ---------------------------------------------------------------------------
# _compute_mismatches
# ---------------------------------------------------------------------------


class TestComputeMismatches:
    def test_mismatches_in_unit_interval(self, ufd, mfd):
        n_samples = 5
        rng = np.random.default_rng(0)
        waveforms = rng.standard_normal(
            (n_samples, len(ufd()))
        ) + 1j * rng.standard_normal((n_samples, len(ufd())))
        polarizations = {"h_plus": waveforms}
        asd = np.ones(len(ufd()))
        mismatches = _compute_mismatches(polarizations, ufd, mfd, asd)
        assert mismatches.shape == (n_samples,)
        assert np.all(mismatches >= 0)
        assert np.all(mismatches <= 1)

    def test_constant_waveform_gives_near_zero_mismatch(self, ufd, mfd):
        # A constant complex waveform is perfectly preserved by decimation + interpolation
        n_samples = 3
        constant = np.ones(len(ufd()), dtype=complex)
        polarizations = {"h_plus": np.stack([constant] * n_samples)}
        asd = np.ones(len(ufd()))
        mismatches = _compute_mismatches(polarizations, ufd, mfd, asd)
        assert np.allclose(mismatches, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _output_settings_path
# ---------------------------------------------------------------------------


class TestOutputSettingsPath:
    def test_replaces_ufd_with_mfd(self):
        path = _output_settings_path("/some/dir/settings_wfd_ufd.yaml")
        assert os.path.basename(path) == "settings_wfd_mfd.yaml"
        assert os.path.dirname(path) == "/some/dir"

    def test_replaces_ufd_suffix_in_longer_name(self):
        path = _output_settings_path("/data/settings_wfd_ufd_32s.yaml")
        assert os.path.basename(path) == "settings_wfd_mfd_32s.yaml"

    def test_appends_mfd_when_ufd_absent(self):
        path = _output_settings_path("/some/dir/settings_wfd.yaml")
        assert os.path.basename(path) == "settings_wfd_mfd.yaml"

    def test_output_in_same_directory_as_input(self):
        path = _output_settings_path("/some/dir/settings_wfd_ufd.yaml")
        assert os.path.dirname(path) == "/some/dir"


# ---------------------------------------------------------------------------
# build_extreme_prior
# ---------------------------------------------------------------------------


class TestBuildExtremePrior:
    def test_chirp_mass_fixed_at_nominal_minimum(self, settings):
        from dingo.gw.prior import build_prior_with_defaults

        nominal_minimum = build_prior_with_defaults(settings["intrinsic_prior"])[
            "chirp_mass"
        ].minimum
        prior = build_extreme_prior(settings)
        assert isinstance(prior["chirp_mass"], DeltaFunction)
        assert prior["chirp_mass"].peak == nominal_minimum

    def test_geocent_time_fixed_at_boundary(self, settings):
        prior = build_extreme_prior(settings)
        assert isinstance(prior["geocent_time"], DeltaFunction)
        assert prior["geocent_time"].peak == 0.12

    def test_input_settings_not_mutated(self, settings):
        original_chirp_mass = settings["intrinsic_prior"]["chirp_mass"]
        original_geocent_time = settings["intrinsic_prior"]["geocent_time"]
        build_extreme_prior(settings)
        assert settings["intrinsic_prior"]["chirp_mass"] == original_chirp_mass
        assert settings["intrinsic_prior"]["geocent_time"] == original_geocent_time

    def test_other_parameters_unchanged(self, settings):
        from dingo.gw.prior import build_prior_with_defaults

        nominal = build_prior_with_defaults(settings["intrinsic_prior"])
        extreme = build_extreme_prior(settings)
        for key in nominal:
            if key in ("chirp_mass", "geocent_time"):
                continue
            assert str(extreme[key]) == str(nominal[key])


# ---------------------------------------------------------------------------
# _evaluate_multibanding_main
# ---------------------------------------------------------------------------

MFD_SETTINGS = {
    "domain": {
        "type": "MultibandedFrequencyDomain",
        "nodes": [20.0, 64.0, 256.0],
        "delta_f_initial": 1.0,
        "base_domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 256.0,
            "delta_f": 1.0,
        },
    },
    "waveform_generator": {
        "approximant": "IMRPhenomPv2",
        "f_ref": 20.0,
        "new_interface": False,
    },
    "intrinsic_prior": INTRINSIC_PRIOR,
}


@pytest.fixture
def mfd_settings_file(tmp_path):
    import yaml

    path = tmp_path / "settings_wfd_mfd.yaml"
    with open(path, "w") as f:
        yaml.dump(MFD_SETTINGS, f)
    return str(path)


class TestEvaluateMultibandingMain:
    def test_raises_for_ufd_settings(self, tmp_path):
        import yaml

        ufd_settings = dict(MFD_SETTINGS)
        ufd_settings["domain"] = {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 256.0,
            "delta_f": 1.0,
        }
        path = str(tmp_path / "settings_ufd.yaml")
        with open(path, "w") as f:
            yaml.dump(ufd_settings, f)
        with pytest.raises(ValueError, match="MultibandedFrequencyDomain"):
            _evaluate_multibanding_main(path, num_samples=2)

    def test_uses_extreme_prior(self, mfd_settings_file):
        """Verify that the extreme prior (min chirp mass, geocent_time=0.12) is applied."""
        captured_prior = {}

        def mock_generate(waveform_generator, prior, num_samples, num_processes):
            captured_prior["prior"] = prior
            ufd = waveform_generator.domain
            n = len(ufd())
            pols = {
                "h_plus": np.ones((num_samples, n), dtype=complex),
                "h_cross": np.ones((num_samples, n), dtype=complex),
            }
            return pd.DataFrame({"dummy": range(num_samples)}), pols

        def mock_parallel(waveform_generator, parameters):
            ufd = waveform_generator.domain
            n = len(ufd())
            return {
                "h_plus": np.ones((len(parameters), n), dtype=complex),
                "h_cross": np.ones((len(parameters), n), dtype=complex),
            }

        with (
            patch(
                "dingo.gw.dataset.evaluate_multibanded_domain.generate_parameters_and_polarizations",
                side_effect=mock_generate,
            ),
            patch(
                "dingo.gw.dataset.evaluate_multibanded_domain.generate_waveforms_parallel",
                side_effect=mock_parallel,
            ),
        ):
            _evaluate_multibanding_main(mfd_settings_file, num_samples=2)

        from dingo.gw.prior import build_prior_with_defaults

        nominal_minimum = build_prior_with_defaults(INTRINSIC_PRIOR)[
            "chirp_mass"
        ].minimum
        prior = captured_prior["prior"]
        assert isinstance(prior["chirp_mass"], DeltaFunction)
        assert prior["chirp_mass"].peak == nominal_minimum
        assert isinstance(prior["geocent_time"], DeltaFunction)
        assert prior["geocent_time"].peak == 0.12
