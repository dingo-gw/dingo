"""Tests for the sampler-implementation switch in the dingo_pipe parser."""

import pytest
from bilby_pipe.utils import parse_args

from dingo.pipe.parser import create_parser


def _parse(tmp_path, extra=()):
    ini = tmp_path / "test.ini"
    ini.write_text("model = model.pt\n")
    args, _ = parse_args([str(ini), *extra], create_parser(top_level=False))
    return args


def test_default_is_legacy(tmp_path):
    assert _parse(tmp_path).sampler_implementation == "legacy"


def test_composed_parses(tmp_path):
    args = _parse(tmp_path, ["--sampler-implementation", "composed"])
    assert args.sampler_implementation == "composed"


def test_invalid_value_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _parse(tmp_path, ["--sampler-implementation", "bogus"])
