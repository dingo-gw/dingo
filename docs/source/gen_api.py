#!/usr/bin/env python3
"""Auto-generate the quartodoc API `sections` in _quarto.yml.

The quartodoc analogue of running ``sphinx-apidoc``: walk the ``dingo`` package,
extract every public class/function from each module (via ``ast``, no import needed),
and list them so quartodoc renders a full-docstring page per object. Objects are
grouped by sub-package, so each appears individually in the API sidebar (one click
from its full docstring, like Sphinx). New objects appear automatically.

Run from ``docs/source``::

    python gen_api.py

then ``quartodoc build`` && ``quarto render`` as usual.
"""
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG_DIR = os.path.join(REPO, "dingo")
QUARTO_YML = os.path.join(HERE, "_quarto.yml")

# Sub-packages to skip entirely (CLI/orchestration layers, not a library API).
SKIP_TOP = {"pipe"}


def discover():
    """Return {group: [dotted.object.path, ...]} of public classes/functions."""
    groups = {}
    for root, dirs, files in os.walk(PKG_DIR):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith((".", "_")) and d != "tests"
        )
        for fname in sorted(files):
            if not fname.endswith(".py") or fname.startswith(("_", "test")):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, PKG_DIR)[:-3].replace(os.sep, ".")  # core.nn.enets
            parts = rel.split(".")
            if parts[0] in SKIP_TOP:
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    tree = ast.parse(fh.read())
            except SyntaxError:
                continue
            objs = [
                f"{rel}.{n.name}"
                for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("_")
            ]
            if not objs:
                continue
            group = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            groups.setdefault(group, []).extend(objs)
    return groups


def build_sections(groups):
    lines = []
    for group in sorted(groups):
        lines.append(f'    - title: "dingo.{group}"')
        lines.append("      contents:")
        for obj in sorted(groups[group]):
            lines.append(f"        - {obj}")
    return "\n".join(lines)


def main():
    groups = discover()
    sections = build_sections(groups)
    with open(QUARTO_YML, encoding="utf-8") as fh:
        text = fh.read()
    new, n_subs = re.subn(
        r"(    # BEGIN AUTOGEN SECTIONS[^\n]*\n).*?(    # END AUTOGEN SECTIONS)",
        lambda m: m.group(1) + sections + "\n" + m.group(2),
        text,
        flags=re.S,
    )
    if n_subs != 1:
        raise SystemExit("AUTOGEN markers not found (or not matched once) in _quarto.yml")
    with open(QUARTO_YML, "w", encoding="utf-8") as fh:
        fh.write(new)
    n_obj = sum(len(v) for v in groups.values())
    print(f"Wrote {n_obj} objects across {len(groups)} groups to _quarto.yml")


if __name__ == "__main__":
    main()
