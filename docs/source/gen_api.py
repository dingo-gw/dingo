#!/usr/bin/env python3
"""Auto-generate the quartodoc API `sections` in _quarto.yml.

The quartodoc analogue of running ``sphinx-apidoc``: walk the ``dingo`` package,
find every public module that defines classes or functions, and list them (grouped
by sub-package) so quartodoc documents all their members. New modules appear
automatically; no hand-maintained object list.

Run from ``docs/source``::

    python gen_api.py

then ``quartodoc build`` && ``quarto render`` as usual.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG_DIR = os.path.join(REPO, "dingo")
QUARTO_YML = os.path.join(HERE, "_quarto.yml")

# Sub-packages to skip entirely (CLI/orchestration layers, not a library API).
SKIP_TOP = {"pipe"}


def discover():
    """Return {group: [module.dotted.path, ...]} for public modules with an API."""
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
            src = open(path, encoding="utf-8", errors="ignore").read()
            if not re.search(r"^(class|def)\s+[A-Za-z]", src, re.M):
                continue  # no public classes/functions -> nothing to document
            group = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            groups.setdefault(group, []).append(rel)
    return groups


def build_sections(groups):
    lines = []
    for group in sorted(groups):
        lines.append(f'    - title: "dingo.{group}"')
        lines.append("      contents:")
        for mod in sorted(groups[group]):
            lines.append(f"        - name: {mod}")
            lines.append("          children: linked")
    return "\n".join(lines)


def main():
    groups = discover()
    sections = build_sections(groups)
    text = open(QUARTO_YML, encoding="utf-8").read()
    new = re.sub(
        r"(    # BEGIN AUTOGEN SECTIONS[^\n]*\n).*?(    # END AUTOGEN SECTIONS)",
        lambda m: m.group(1) + sections + "\n" + m.group(2),
        text,
        flags=re.S,
    )
    if new == text or "BEGIN AUTOGEN" not in new:
        raise SystemExit("AUTOGEN markers not found in _quarto.yml")
    open(QUARTO_YML, "w", encoding="utf-8").write(new)
    n_mods = sum(len(v) for v in groups.values())
    print(f"Wrote {n_mods} modules across {len(groups)} groups to _quarto.yml")


if __name__ == "__main__":
    main()
