"""Tests for the fixed-context-parameters option in the dingo_pipe parser."""

from bilby_pipe.utils import convert_string_to_dict, parse_args

from dingo.pipe.parser import create_parser

DEMO_FORM = "{chirp_mass_proxy: 1.19786, ra: 3.44616, dec: -0.408084}"
DEMO_DICT = {"chirp_mass_proxy": 1.19786, "ra": 3.44616, "dec": -0.408084}


def _parse(tmp_path, ini_body="model = model.pt\n", extra=()):
    ini = tmp_path / "test.ini"
    ini.write_text(ini_body)
    args, _ = parse_args([str(ini), *extra], create_parser(top_level=False))
    return args


def test_default_is_none(tmp_path):
    assert _parse(tmp_path).fixed_context_parameters is None


def test_cli_dict_form_parses(tmp_path):
    args = _parse(tmp_path, extra=["--fixed-context-parameters", DEMO_FORM])
    assert convert_string_to_dict(args.fixed_context_parameters) == DEMO_DICT


def test_ini_dict_form_parses(tmp_path):
    # The published binary-neutron-star-demo INI form, read from the INI file
    # and converted the way SamplingInput does.
    args = _parse(
        tmp_path,
        ini_body=f"model = model.pt\nfixed-context-parameters = {DEMO_FORM}\n",
    )
    assert convert_string_to_dict(args.fixed_context_parameters) == DEMO_DICT
