import types

import pytest

from dingo.pipe.pp_test import (
    create_parser,
    get_basename,
    get_results_filenames,
)


def test_get_basename_defaults_outdir_to_directory():
    args = types.SimpleNamespace(outdir=None, directory="/tmp/foo", label=None)
    assert get_basename(args) == "/tmp/foo/"


def test_get_basename_includes_label():
    args = types.SimpleNamespace(outdir="/tmp/out", directory="/tmp/foo", label="run")
    assert get_basename(args) == "/tmp/out/run_"


def test_get_results_filenames_prefers_importance_sampling(tmp_path):
    for name in (
        "a_importance_sampling.hdf5",
        "b_importance_sampling.hdf5",
        "c_sampling.hdf5",
    ):
        (tmp_path / name).touch()
    args = types.SimpleNamespace(directory=str(tmp_path), filter=None, n=None)
    # When importance_sampling results exist, plain sampling results are ignored.
    assert len(get_results_filenames(args)) == 2


def test_get_results_filenames_truncates_to_n(tmp_path):
    for name in ("a_importance_sampling.hdf5", "b_importance_sampling.hdf5"):
        (tmp_path / name).touch()
    args = types.SimpleNamespace(directory=str(tmp_path), filter=None, n=1)
    assert len(get_results_filenames(args)) == 1


def test_get_results_filenames_empty_raises(tmp_path):
    args = types.SimpleNamespace(directory=str(tmp_path), filter=None, n=None)
    with pytest.raises(FileNotFoundError, match="No results"):
        get_results_filenames(args)


def test_pp_test_create_parser_builds():
    assert create_parser() is not None
