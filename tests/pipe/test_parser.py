from bilby_pipe.bilbyargparser import BilbyArgParser

from dingo.pipe.parser import create_parser
from dingo.pipe.plot import create_parser as create_plot_parser


def test_create_parser_builds():
    """Smoke test: create_parser() constructs the dingo_pipe argument parser,
    exercising the many add_argument definitions without error."""
    parser = create_parser()
    assert isinstance(parser, BilbyArgParser)


def test_create_plot_parser_builds():
    assert create_plot_parser() is not None
