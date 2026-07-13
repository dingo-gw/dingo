# Documentation

The documentation is built with [Quarto](https://quarto.org) +
[quartodoc](https://machow.github.io/quartodoc/). Source lives in `source/`.

## Prerequisites

- [Quarto](https://quarto.org/docs/download/) — a standalone binary.
- Python doc tooling: from the repo root, `pip install --group docs jupyter`
  (quartodoc/griffe, pinned; `jupyter` is needed to render the notebook pages).
  The `dingo` package must be importable / on the path; the API build parses it
  statically, so its full runtime is not required.

## Building

From `source/`:

```sh
python gen_api.py     # refresh the API object list in _quarto.yml (the sphinx-apidoc analogue)
python build_api.py   # generate reference/*.qmd from docstrings (merges __init__ params)
quarto render         # build the site into _site/   (or: quarto preview)
```

The rendered site lands in `source/_site/` (git-ignored). Use `quarto preview` for a
live-reloading local preview.

## Deployment

The docs are hosted on [Read the Docs](https://app.readthedocs.org/projects/dingo-gw/)
at https://dingo-gw.readthedocs.io, which runs this same build via `.readthedocs.yaml`
(Quarto has no native Read the Docs support, so `build.commands` overrides the build).
Every pull request gets a preview build, linked from the PR checks; released versions
can be activated for the version switcher in the Read the Docs dashboard.
