import numpy as np

from dingo.core.utils.misc import get_version, recursive_check_dicts_are_equal


def test_get_version_returns_string():
    version = get_version()
    assert isinstance(version, str) and version


def test_recursive_check_equal_nested_dicts():
    a = {"x": 1, "nested": {"y": 2}}
    assert recursive_check_dicts_are_equal(a, {"x": 1, "nested": {"y": 2}}) is True


def test_recursive_check_different_keys():
    assert recursive_check_dicts_are_equal({"a": 1}, {"b": 1}) is False


def test_recursive_check_different_types():
    # 1 (int) vs 1.0 (float) differ by type.
    assert recursive_check_dicts_are_equal({"a": 1}, {"a": 1.0}) is False


def test_recursive_check_nested_value_differs():
    assert recursive_check_dicts_are_equal({"n": {"y": 2}}, {"n": {"y": 3}}) is False


def test_recursive_check_arrays():
    assert recursive_check_dicts_are_equal(
        {"a": np.array([1, 2])}, {"a": np.array([1, 2])}
    )
    assert not recursive_check_dicts_are_equal(
        {"a": np.array([1, 2])}, {"a": np.array([1, 3])}
    )


def test_recursive_check_string_numbers_within_tolerance():
    # Numeric literals in strings are compared with a tolerance (they can drift by
    # float-precision across machines); non-numeric characters must match exactly.
    a = {"p": "Uniform(minimum=1.0000000, maximum=2.0)"}
    b = {"p": "Uniform(minimum=1.0000001, maximum=2.0)"}
    assert recursive_check_dicts_are_equal(a, b) is True


def test_recursive_check_string_numbers_beyond_tolerance():
    a = {"p": "Uniform(minimum=1.0, maximum=2.0)"}
    b = {"p": "Uniform(minimum=9.0, maximum=2.0)"}
    assert recursive_check_dicts_are_equal(a, b) is False


def test_recursive_check_string_non_numeric_differs():
    a = {"p": "Uniform(minimum=1.0)"}
    b = {"p": "Normal(minimum=1.0)"}
    assert recursive_check_dicts_are_equal(a, b) is False
