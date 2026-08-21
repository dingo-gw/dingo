"""
Regression tests for the deprecated dataset-level legacy wrappers.

The historic ``dingo.gw.dataset.generate_dataset`` module is now a shim; the
tests below verify the shim resolves the historic names and each emits a
DeprecationWarning.
"""

import pytest


def test_shim_module_resolves():
    """Historic import path still works."""
    import dingo.gw.dataset.generate_dataset as legacy

    assert hasattr(legacy, "generate_dataset")
    assert hasattr(legacy, "generate_parameters_and_polarizations")
    assert hasattr(legacy, "_generate_dataset_main")


def test_generate_dataset_deprecated():
    from dingo.gw.dataset.generate_dataset import generate_dataset

    with pytest.raises((ValueError, TypeError, KeyError)):
        with pytest.warns(DeprecationWarning, match="legacy entrypoint"):
            # Call with invalid settings to trigger the warning without
            # running heavy generation; we only assert the warning fires.
            generate_dataset({}, num_processes=1)


def test_generate_parameters_and_polarizations_deprecated():
    from dingo.gw.dataset.generate_dataset import (
        generate_parameters_and_polarizations,
    )

    with pytest.raises(Exception):
        with pytest.warns(DeprecationWarning, match="stacked arrays"):
            # Same idea: pass empty inputs, only check warning path.
            generate_parameters_and_polarizations(
                object(), object(), num_samples=0, num_processes=1
            )


def test_generate_dataset_main_deprecated():
    from dingo.gw.dataset.generate_dataset import _generate_dataset_main

    with pytest.raises(Exception):
        with pytest.warns(DeprecationWarning, match="CLI helper"):
            _generate_dataset_main("/nonexistent/settings.yaml", "/tmp/out.hdf5", 1)
