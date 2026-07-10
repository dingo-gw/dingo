import pytest

from dingo.core.utils.backward_compatibility import check_minimum_version


def test_check_minimum_version_passes_for_recent_version():
    # A version well above the compatibility threshold must not raise.
    assert check_minimum_version("dingo=99.0.0", raise_exception=True) is None


def test_check_minimum_version_raises_for_old_version():
    with pytest.raises(ValueError, match="backwards compatibility"):
        check_minimum_version("dingo=0.0.1", raise_exception=True)


def test_check_minimum_version_treats_none_as_oldest():
    # A "None" version string is treated as 0.0.0, i.e. below the threshold.
    with pytest.raises(ValueError):
        check_minimum_version("None", raise_exception=True)


def test_check_minimum_version_warns_but_does_not_raise_by_default():
    # With raise_exception=False (default), an old version only warns.
    assert check_minimum_version("dingo=0.0.1") is None
