#!/usr/bin/env python3
"""Build the quartodoc API reference, merging ``__init__`` params into the class.

Why this exists
---------------
Dingo documents constructor arguments in the ``__init__`` docstring (a numpydoc
``Parameters`` block on ``__init__``), which is the convention the whole codebase
follows. Sphinx + napoleon merges that ``__init__`` docstring into the class page,
so readthedocs shows a full Parameters table for every class. quartodoc reads *only*
the class docstring, so those tables silently vanish and the class pages look bare.

mkdocstrings-python solves this with a built-in ``merge_init_into_class`` option;
quartodoc (0.11) has no such knob. This script adds the equivalent as a tiny griffe
post-processing shim: after quartodoc loads each object, if it is a class whose
``__init__`` carries a numpydoc ``Parameters`` section (and the class docstring does
not already have one), we append that section to the class docstring. griffe then
parses and renders it as the usual Parameters table. No dingo source changes; this
fixes every class at once.

Everything else is stock ``quartodoc build`` — we just wrap the object loader.

Usage (replaces the bare ``quartodoc build`` step)::

    python gen_api.py       # regenerate the sections list in _quarto.yml
    python build_api.py     # build reference/*.qmd (this script)
    quarto render           # render the site

Run from ``docs/source``.
"""
import os
import re
import sys

# Make the `dingo` package importable when run from docs/source (repo root holds it).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import quartodoc.builder.blueprint as _bp  # noqa: E402
from griffe import Docstring  # noqa: E402
from quartodoc import Builder  # noqa: E402

# A numpydoc "Parameters" section header:  Parameters\n----------
_PARAMS_HEADER = re.compile(r"(?:^|\n)[ \t]*Parameters[ \t]*\n[ \t]*-{3,}", re.M)

_orig_get_object = _bp._get_object


def _merge_init_params(obj):
    """If ``obj`` is a class documenting its params on ``__init__``, lift them."""
    try:
        if not getattr(obj, "is_class", False):
            return obj
        init = obj.members.get("__init__")
        if init is None or init.docstring is None:
            return obj
        add = init.docstring.value or ""
        if not _PARAMS_HEADER.search(add):
            return obj  # __init__ has no Parameters section to lift
        cls_doc = obj.docstring.value if (obj.docstring and obj.docstring.value) else ""
        if _PARAMS_HEADER.search(cls_doc):
            return obj  # class docstring already documents its parameters
        if not cls_doc.strip():
            obj.docstring = Docstring(
                add,
                parent=obj,
                parser=init.docstring.parser,
                parser_options=init.docstring.parser_options,
            )
        else:
            obj.docstring.value = cls_doc.rstrip() + "\n\n" + add
    except Exception:
        # Never let the shim break a build; fall back to quartodoc's default output.
        pass
    return obj


def _patched_get_object(path, **kwargs):
    return _merge_init_params(_orig_get_object(path, **kwargs))


# Intercept the loader used by BlueprintTransformer (blueprint.py binds the module
# global `_get_object` into a partial at build time, so reassigning it here suffices).
_bp._get_object = _patched_get_object


def main():
    builder = Builder.from_quarto_config("_quarto.yml")
    builder.build()


if __name__ == "__main__":
    main()
