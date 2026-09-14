import types

from bilby_pipe.utils import convert_string_to_dict

from dingo.pipe.utils import dict_to_string


def test_dict_to_string_round_trips_through_bilby_pipe_parser():
    """dingo's dict_to_string is the inverse of bilby_pipe's convert_string_to_dict.

    Inspired by bilby_pipe's tests/utils_test.py::test_dict_converter. dingo writes
    settings dicts into INI strings via dict_to_string and reads them back with
    bilby_pipe's convert_string_to_dict, so the round-trip must be faithful.
    """
    d = {"num_samples": 1000, "batch_size": 50, "label": "run"}
    parsed = convert_string_to_dict(dict_to_string(d))
    assert parsed == d


def test_strip_unwanted_submission_keys():
    from dingo.pipe.utils import _strip_unwanted_submission_keys

    job = types.SimpleNamespace(
        getenv="x",
        universe="vanilla",
        extra_lines=[
            "priority = 10",
            "accounting_group = ligo.dev",
            "ENV GET HTGETTOKENOPTS foo",
            "request_memory = 4GB",
        ],
    )
    _strip_unwanted_submission_keys(job)
    assert job.getenv is None
    assert job.universe is None
    # priority / accounting_group / HTGETTOKENOPTS lines are removed; others kept.
    assert job.extra_lines == ["request_memory = 4GB"]
