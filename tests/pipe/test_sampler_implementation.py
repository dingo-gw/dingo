"""Tests for the sampler-implementation switch in the dingo_pipe parser."""

import pytest
from bilby_pipe.utils import parse_args

from dingo.pipe.parser import create_parser


def _parse(tmp_path, extra=()):
    ini = tmp_path / "test.ini"
    ini.write_text("model = model.pt\n")
    args, _ = parse_args([str(ini), *extra], create_parser(top_level=False))
    return args


def test_default_is_composed(tmp_path):
    assert _parse(tmp_path).sampler_implementation == "composed"


def test_legacy_still_parses(tmp_path):
    # Accepted at parse time so that old INI files fail at sampler construction
    # with a clear pointer to the legacy-samplers-final tag, not an argparse error.
    args = _parse(tmp_path, ["--sampler-implementation", "legacy"])
    assert args.sampler_implementation == "legacy"


def test_invalid_value_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _parse(tmp_path, ["--sampler-implementation", "bogus"])
