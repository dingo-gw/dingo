import types
from unittest import mock

import numpy as np
import pytest
from bilby_pipe.utils import BilbyPipeError

import dingo.pipe.dag_creator as dag_creator
from dingo.pipe.dag_creator import (
    generate_dag,
    get_parallel_list,
    get_trigger_time_list,
)


def _inputs(**overrides):
    defaults = dict(
        gaussian_noise=False,
        zero_noise=False,
        trigger_time=None,
        model_reference_time=100.0,
        n_simulation=3,
        gpstimes=None,
        duration=8.0,
        post_trigger_duration=2.0,
        # generate_dag orchestration flags
        simple_submission=False,
        importance_sample=False,
        importance_sampling_updates={},
        n_parallel=1,
        plot_node_needed=False,
        plot_pp=False,
        create_summary=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_trigger_times_simulation_uses_reference_time():
    # Simulated noise without a trigger time -> model_reference_time per simulation.
    assert get_trigger_time_list(_inputs(gaussian_noise=True)) == [100.0, 100.0, 100.0]


def test_trigger_times_simulation_with_fixed_trigger_time():
    times = get_trigger_time_list(_inputs(zero_noise=True, trigger_time=50.0))
    assert times == [50.0, 50.0, 50.0]


def test_trigger_times_real_event():
    assert get_trigger_time_list(_inputs(trigger_time=1126259462.4)) == [1126259462.4]


def test_trigger_times_from_gpstimes():
    # trigger = gps_start + duration - post_trigger_duration.
    times = get_trigger_time_list(_inputs(gpstimes=np.array([1000.0, 2000.0])))
    np.testing.assert_array_equal(times, [1006.0, 2006.0])


def test_trigger_times_undetermined_raises():
    with pytest.raises(BilbyPipeError):
        get_trigger_time_list(_inputs())


def test_get_parallel_list_single():
    assert get_parallel_list(types.SimpleNamespace(n_parallel=1)) == [""]


def test_get_parallel_list_multiple():
    assert get_parallel_list(types.SimpleNamespace(n_parallel=3)) == [
        "part0",
        "part1",
        "part2",
    ]


# ---------------------------------------------------------------------------
# generate_dag control flow (node classes mocked out)
# ---------------------------------------------------------------------------

_NODE_CLASSES = [
    "Dag",
    "GenerationNode",
    "SamplingNode",
    "ImportanceSamplingNode",
    "MergeNode",
    "PlotNode",
    "PlotPPNode",
    "PESummaryNode",
]


def _run_generate_dag(inputs):
    """Run generate_dag with all node/Dag classes replaced by mocks, returning the
    dict of mocks so call counts can be inspected."""
    with mock.patch.multiple(
        dag_creator, **{name: mock.DEFAULT for name in _NODE_CLASSES}
    ) as mocks:
        generate_dag(inputs)
    return mocks


def test_generate_dag_creates_generation_and_sampling_nodes():
    mocks = _run_generate_dag(_inputs(trigger_time=1126259462.4))
    # One real event -> one generation and one sampling node; the DAG is built.
    assert mocks["GenerationNode"].call_count == 1
    assert mocks["SamplingNode"].call_count == 1
    assert mocks["ImportanceSamplingNode"].call_count == 0
    assert mocks["Dag"].return_value.build.called


def test_generate_dag_one_node_per_simulation():
    mocks = _run_generate_dag(
        _inputs(gaussian_noise=True, trigger_time=None, n_simulation=3)
    )
    assert mocks["GenerationNode"].call_count == 3
    assert mocks["SamplingNode"].call_count == 3


def test_generate_dag_importance_sampling_single_parallel_no_merge():
    mocks = _run_generate_dag(
        _inputs(trigger_time=1126259462.4, importance_sample=True, n_parallel=1)
    )
    assert mocks["ImportanceSamplingNode"].call_count == 1
    assert mocks["MergeNode"].call_count == 0


def test_generate_dag_importance_sampling_parallel_merges():
    mocks = _run_generate_dag(
        _inputs(trigger_time=1126259462.4, importance_sample=True, n_parallel=3)
    )
    # 3 parallel importance-sampling jobs, recombined by a single merge node.
    assert mocks["ImportanceSamplingNode"].call_count == 3
    assert mocks["MergeNode"].call_count == 1


def test_generate_dag_creates_plot_and_summary_nodes_when_requested():
    mocks = _run_generate_dag(
        _inputs(
            trigger_time=1126259462.4,
            plot_node_needed=True,
            plot_pp=True,
            create_summary=True,
        )
    )
    assert mocks["PlotNode"].call_count == 1
    assert mocks["PlotPPNode"].call_count == 1
    assert mocks["PESummaryNode"].call_count == 1
