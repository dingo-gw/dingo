"""Guardrails for dingo_pipe staying compatible with the bilby_pipe API.

dingo keeps its own copy of the bilby_pipe argument parser and subclasses
bilby_pipe's ``DataGenerationInput``. When bilby_pipe adds an argument that its
data-generation code reads, dingo's parser must define it too, otherwise runs
fail at generation time with an ``AttributeError`` on the args namespace.
"""

import inspect
import re
from argparse import Namespace

from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput

from dingo.pipe.data_generation import DataGenerationInput, create_generation_parser


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


def test_datageneration_input_provides_bilby_injection_attrs(tmp_path):
    """dingo's DataGenerationInput overrides bilby_pipe's __init__, so it must set the
    self.* attributes bilby_pipe reads when building an injection. Calling the helper
    that failed in a real injection run raises AttributeError if dingo omits one such
    attribute (e.g. waveform_generator_class_ctor_args)."""
    args = {a.dest: a.default for a in create_generation_parser()._actions if a.dest != "help"}
    args.update(
        model="dummy.pt", model_init=None, idx=0, label="test", outdir=str(tmp_path),
        trigger_time=0.0, detectors=["H1", "L1"], duration=4.0, sampling_frequency=4096,
        minimum_frequency=20.0, maximum_frequency=1024.0, reference_frequency=20.0,
    )
    inputs = DataGenerationInput(Namespace(**args), [], create_data=False)
    # create_data (skipped) normally sets this; None exercises the fallback branch.
    inputs.injection_waveform_generator_class_ctor_args = None
    assert inputs.get_default_injection_waveform_generator_class_ctor_arguments() == {}
