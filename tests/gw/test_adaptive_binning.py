"""
Tests for the dingo.gw.domains.binning.adaptive_binning module.

Ported from dingo-waveform tests/test_adaptive_binning.py.
"""

import numpy as np
import pytest

from dingo.gw.domains.binning.adaptive_binning import (
    Band,
    BinningParameters,
    compute_adaptive_binning,
    compile_binning_from_bands,
    decimate,
    decimate_uniform,
    plan_bands,
    _infer_base_offset_idx_auto,
)


# ==============================================================================
# Test Band dataclass
# ==============================================================================


def test_band_creation():
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=200,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=50,
        remainder=0,
        bin_start=0,
        bin_end=50,
    )

    assert band.index == 0
    assert band.node_lower == 20.0
    assert band.node_upper == 40.0
    assert band.delta_f_band == 0.5
    assert band.decimation_factor_band == 2
    assert band.num_bins == 50


def test_band_immutability():
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=200,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=50,
        remainder=0,
        bin_start=0,
        bin_end=50,
    )

    with pytest.raises(Exception):  # FrozenInstanceError
        band.index = 1


def test_band_properties():
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=200,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=50,
        remainder=0,
        bin_start=0,
        bin_end=50,
    )

    assert band.bin_slice == slice(0, 50)
    assert band.band_width_indices == 100
    assert band.covered_base_samples == 100  # 50 * 2
    assert band.coverage_ratio == 1.0  # 100 / 100


def test_band_coverage_ratio_with_remainder():
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=40.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=203,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=51,
        remainder=1,
        bin_start=0,
        bin_end=51,
    )

    assert band.band_width_indices == 103
    assert band.covered_base_samples == 102  # 51 * 2
    assert pytest.approx(band.coverage_ratio, rel=1e-6) == 102 / 103


def test_band_coverage_ratio_zero_width():
    band = Band(
        index=0,
        node_lower=20.0,
        node_upper=20.0,
        node_lower_idx=100,
        node_upper_idx_exclusive=100,
        delta_f_band=0.5,
        decimation_factor_band=2,
        num_bins=0,
        remainder=0,
        bin_start=0,
        bin_end=0,
    )

    assert band.coverage_ratio == 0.0


# ==============================================================================
# Test plan_bands()
# ==============================================================================


def test_plan_bands_basic():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    assert len(bands) == 2
    assert bands[0].index == 0
    assert bands[1].index == 1

    assert bands[0].node_lower == 20.0
    assert bands[0].node_upper == 40.0
    assert bands[0].delta_f_band == 0.25
    assert bands[0].decimation_factor_band == 1

    assert bands[1].node_lower == 40.0
    assert bands[1].node_upper == 80.0
    assert bands[1].delta_f_band == 0.5
    assert bands[1].decimation_factor_band == 2


def test_plan_bands_dyadic_spacing():
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    assert len(bands) == 3
    assert bands[0].delta_f_band == 0.25
    assert bands[1].delta_f_band == 0.5
    assert bands[2].delta_f_band == 1.0
    assert bands[0].decimation_factor_band == 2
    assert bands[1].decimation_factor_band == 4
    assert bands[2].decimation_factor_band == 8


def test_plan_bands_single_band():
    nodes = [20.0, 100.0]
    base_delta_f = 0.5
    delta_f_initial = 0.5

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    assert len(bands) == 1
    assert bands[0].index == 0
    assert bands[0].node_lower == 20.0
    assert bands[0].node_upper == 100.0


def test_plan_bands_contiguous_bin_ranges():
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.125

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)

    for i in range(1, len(bands)):
        assert bands[i].bin_start == bands[i - 1].bin_end


def test_plan_bands_invalid_base_delta_f():
    nodes = [20.0, 40.0]

    with pytest.raises(ValueError, match="base_delta_f must be positive"):
        plan_bands(nodes, base_delta_f=0.0, delta_f_initial=0.25)

    with pytest.raises(ValueError, match="base_delta_f must be positive"):
        plan_bands(nodes, base_delta_f=-0.25, delta_f_initial=0.25)


def test_plan_bands_invalid_delta_f_initial():
    nodes = [20.0, 40.0]

    with pytest.raises(ValueError, match="delta_f_initial must be positive"):
        plan_bands(nodes, base_delta_f=0.25, delta_f_initial=0.0)

    with pytest.raises(ValueError, match="delta_f_initial must be positive"):
        plan_bands(nodes, base_delta_f=0.25, delta_f_initial=-0.25)


def test_plan_bands_invalid_nodes_shape():
    nodes_2d = [[20.0, 40.0], [60.0, 80.0]]

    with pytest.raises(ValueError, match="Expected 1D nodes array"):
        plan_bands(nodes_2d, base_delta_f=0.25, delta_f_initial=0.25)


def test_plan_bands_too_few_nodes():
    with pytest.raises(ValueError, match="at least two elements"):
        plan_bands([20.0], base_delta_f=0.25, delta_f_initial=0.25)

    with pytest.raises(ValueError, match="at least two elements"):
        plan_bands([], base_delta_f=0.25, delta_f_initial=0.25)


def test_plan_bands_non_increasing_nodes():
    with pytest.raises(ValueError, match="must be strictly increasing"):
        plan_bands([40.0, 40.0], base_delta_f=0.25, delta_f_initial=0.25)

    with pytest.raises(ValueError, match="must be strictly increasing"):
        plan_bands([40.0, 20.0], base_delta_f=0.25, delta_f_initial=0.25)


def test_plan_bands_decimation_too_small():
    nodes = [20.0, 40.0]
    base_delta_f = 0.5
    delta_f_initial = 0.25

    with pytest.raises(ValueError, match="Invalid decimation factors"):
        plan_bands(nodes, base_delta_f, delta_f_initial)


# ==============================================================================
# Test compile_binning_from_bands()
# ==============================================================================


def test_compile_binning_from_bands_basic():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    assert isinstance(params, BinningParameters)
    assert params.num_bands == 2
    assert params.total_bins > 0
    assert params.base_delta_f == base_delta_f
    assert params.delta_f_initial == delta_f_initial


def test_compile_binning_reconstructs_nodes():
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    np.testing.assert_allclose(params.nodes, nodes, rtol=1e-6)


def test_compile_binning_per_bin_arrays():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    assert len(params.band_assignment) == params.total_bins
    assert len(params.delta_f) == params.total_bins
    assert len(params.f_base_lower) == params.total_bins
    assert len(params.f_base_upper) == params.total_bins
    assert len(params.base_lower_idx) == params.total_bins
    assert len(params.base_upper_idx) == params.total_bins


def test_compile_binning_per_band_arrays():
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    assert len(params.delta_f_bands) == params.num_bands
    assert len(params.decimation_factors_bands) == params.num_bands
    assert len(params.num_bins_bands) == params.num_bands
    assert len(params.remainder_per_band) == params.num_bands
    assert params.band_bin_ranges.shape == (params.num_bands, 2)


def test_compile_binning_empty_bands():
    params = compile_binning_from_bands([], base_delta_f=0.25, delta_f_initial=0.25)

    assert params.num_bands == 0
    assert params.total_bins == 0
    assert len(params.nodes) == 0
    assert len(params.band_assignment) == 0


def test_compile_binning_invalid_band_indices():
    band0 = Band(
        index=0, node_lower=20.0, node_upper=40.0,
        node_lower_idx=80, node_upper_idx_exclusive=160,
        delta_f_band=0.25, decimation_factor_band=1,
        num_bins=80, remainder=0, bin_start=0, bin_end=80,
    )
    band2 = Band(
        index=2, node_lower=40.0, node_upper=80.0,
        node_lower_idx=160, node_upper_idx_exclusive=320,
        delta_f_band=0.5, decimation_factor_band=2,
        num_bins=80, remainder=0, bin_start=80, bin_end=160,
    )

    with pytest.raises(ValueError, match="indexed 0..num_bands-1 without gaps"):
        compile_binning_from_bands([band0, band2], base_delta_f=0.25, delta_f_initial=0.25)


def test_compile_binning_non_contiguous_boundaries():
    band0 = Band(
        index=0, node_lower=20.0, node_upper=40.0,
        node_lower_idx=80, node_upper_idx_exclusive=160,
        delta_f_band=0.25, decimation_factor_band=1,
        num_bins=80, remainder=0, bin_start=0, bin_end=80,
    )
    band1 = Band(
        index=1, node_lower=50.0, node_upper=80.0,
        node_lower_idx=200, node_upper_idx_exclusive=320,
        delta_f_band=0.5, decimation_factor_band=2,
        num_bins=60, remainder=0, bin_start=80, bin_end=140,
    )

    with pytest.raises(ValueError, match="Non-contiguous band boundaries"):
        compile_binning_from_bands([band0, band1], base_delta_f=0.25, delta_f_initial=0.25)


# ==============================================================================
# Test compute_adaptive_binning()
# ==============================================================================


def test_compute_adaptive_binning_basic():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    assert isinstance(params, BinningParameters)
    assert params.num_bands == 2
    assert params.total_bins > 0


def test_compute_adaptive_binning_matches_two_step():
    nodes = [20.0, 40.0, 80.0, 160.0]
    base_delta_f = 0.125
    delta_f_initial = 0.25

    bands = plan_bands(nodes, base_delta_f, delta_f_initial)
    params1 = compile_binning_from_bands(bands, base_delta_f, delta_f_initial)

    params2 = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    assert params1.num_bands == params2.num_bands
    assert params1.total_bins == params2.total_bins
    np.testing.assert_array_equal(params1.nodes, params2.nodes)
    np.testing.assert_array_equal(params1.num_bins_bands, params2.num_bins_bands)


# ==============================================================================
# Test decimate_uniform()
# ==============================================================================


def test_decimate_uniform_pick_policy():
    data = np.arange(100, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    expected = data[::5]
    np.testing.assert_array_equal(result, expected)
    assert result.shape[-1] == 20


def test_decimate_uniform_mean_policy():
    data = np.arange(100, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="mean")

    expected = np.array([2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0,
                         52.0, 57.0, 62.0, 67.0, 72.0, 77.0, 82.0, 87.0, 92.0, 97.0])
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_decimate_uniform_with_remainder():
    data = np.arange(103, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    assert result.shape[-1] == 20
    expected = data[:100:5]
    np.testing.assert_array_equal(result, expected)


def test_decimate_uniform_multidimensional():
    data = np.arange(200, dtype=np.float32).reshape(2, 100)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    assert result.shape == (2, 20)
    np.testing.assert_array_equal(result[0], data[0, ::5])
    np.testing.assert_array_equal(result[1], data[1, ::5])


def test_decimate_uniform_factor_one():
    data = np.arange(100, dtype=np.float32)

    result = decimate_uniform(data, decimation_factor=1, policy="pick")

    np.testing.assert_array_equal(result, data)


def test_decimate_uniform_empty_result():
    data = np.arange(3, dtype=np.float32)
    decimation_factor = 5

    result = decimate_uniform(data, decimation_factor, policy="pick")

    assert result.shape[-1] == 0


def test_decimate_uniform_invalid_factor():
    data = np.arange(100, dtype=np.float32)

    with pytest.raises(ValueError, match="decimation_factor must be >= 1"):
        decimate_uniform(data, decimation_factor=0)


# ==============================================================================
# Test decimate()
# ==============================================================================


def test_decimate_explicit_mode():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, base_offset_idx=0, mode="explicit", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_auto_mode_full_grid():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1]) + 100
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, mode="auto", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_auto_mode_windowed():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    coverage = int(params.nodes_indices[-1] - params.nodes_indices[0])
    data = np.arange(coverage, dtype=np.float32)

    result = decimate(data, params, mode="auto", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_multidimensional():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(2 * data_len, dtype=np.float32).reshape(2, data_len)

    result = decimate(data, params, base_offset_idx=0, mode="explicit", policy="pick")

    assert result.shape == (2, params.total_bins)


def test_decimate_with_positive_offset():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    offset = 50
    data_len = int(params.nodes_indices[-1]) + offset
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, base_offset_idx=offset, mode="explicit", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_instance_method():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(data_len, dtype=np.float32)

    result = params.decimate(data, base_offset_idx=0, mode="explicit", policy="pick")

    assert result.shape[-1] == params.total_bins


def test_decimate_out_of_bounds():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data = np.arange(10, dtype=np.float32)

    with pytest.raises(IndexError, match="Input slice out of bounds"):
        decimate(data, params, base_offset_idx=0, mode="explicit", check=True)


def test_decimate_check_disabled():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.arange(data_len, dtype=np.float32)

    result = decimate(data, params, base_offset_idx=0, mode="explicit", check=False)
    assert result.shape[-1] == params.total_bins


# ==============================================================================
# Test _infer_base_offset_idx_auto()
# ==============================================================================


def test_infer_offset_windowed_coverage():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    coverage = int(params.nodes_indices[-1] - params.nodes_indices[0])
    offset = _infer_base_offset_idx_auto(params, coverage)

    expected_offset = -int(params.nodes_indices[0])
    assert offset == expected_offset


def test_infer_offset_full_grid():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1]) + 100
    offset = _infer_base_offset_idx_auto(params, data_len)

    assert offset == 0


def test_infer_offset_ambiguous():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    ambiguous_len = int(params.nodes_indices[-1] - params.nodes_indices[0]) + 50

    with pytest.raises(ValueError, match="Cannot auto-infer base_offset_idx"):
        _infer_base_offset_idx_auto(params, ambiguous_len)


# ==============================================================================
# Integration tests
# ==============================================================================


def test_end_to_end_simple_binning():
    nodes = [20.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    waveform = np.sin(np.linspace(0, 10 * np.pi, data_len)).astype(np.float32)

    decimated = params.decimate(waveform, base_offset_idx=0, mode="explicit")

    assert decimated.shape[-1] == params.total_bins
    assert len(decimated) == params.total_bins


def test_end_to_end_multibanded():
    nodes = [20.0, 26.0, 34.0, 46.0, 62.0, 78.0, 1038.0]
    base_delta_f = 0.0625
    delta_f_initial = 0.0625

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    assert params.num_bands == 6

    for i in range(params.num_bands):
        expected_delta_f = delta_f_initial * (2 ** i)
        assert pytest.approx(params.delta_f_bands[i], rel=1e-6) == expected_delta_f


def test_consistency_between_pick_and_mean():
    nodes = [0.0, 40.0, 80.0]
    base_delta_f = 0.25
    delta_f_initial = 0.25

    params = compute_adaptive_binning(nodes, delta_f_initial, base_delta_f)

    data_len = int(params.nodes_indices[-1])
    data = np.sin(np.linspace(0, 2 * np.pi, data_len)).astype(np.float32)

    result_pick = params.decimate(data, base_offset_idx=0, policy="pick")
    result_mean = params.decimate(data, base_offset_idx=0, policy="mean")

    assert result_pick.shape == result_mean.shape
