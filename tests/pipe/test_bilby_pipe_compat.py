"""Guardrails for dingo_pipe staying compatible with the bilby_pipe API.

dingo keeps its own copy of the bilby_pipe argument parser and subclasses
bilby_pipe's ``DataGenerationInput``. When bilby_pipe adds an argument that its
data-generation code reads, dingo's parser must define it too, otherwise runs
fail at generation time with an ``AttributeError`` on the args namespace.
"""

import inspect
import re

from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput

from dingo.pipe.data_generation import create_generation_parser


def _args_attributes_read_in(method):
    """Names accessed as ``args.<name>`` in the source of ``method``."""
    source = inspect.getsource(method)
    return set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)", source))


def test_generation_parser_provides_bilby_create_data_args():
    """Every ``args.<name>`` read by bilby_pipe's ``DataGenerationInput.create_data``
    -- the code path a dingo_pipe injection exercises at generation time -- must be
    defined by dingo's generation parser."""
    needed = _args_attributes_read_in(BilbyDataGenerationInput.create_data)
    provided = {action.dest for action in create_generation_parser()._actions}
    missing = needed - provided
    assert not missing, (
        "bilby_pipe's DataGenerationInput.create_data reads arguments that dingo's "
        f"generation parser does not define: {sorted(missing)}. Add them to "
        "create_parser() in dingo/pipe/parser.py (mirroring bilby_pipe), or update "
        "dingo for the new bilby_pipe API."
    )
