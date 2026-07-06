# Documentation

The documentation is built with [Quarto](https://quarto.org) +
[quartodoc](https://machow.github.io/quartodoc/). Source lives in `source/`.

## Prerequisites

- [Quarto](https://quarto.org/docs/download/) — a standalone binary.
- Python doc tooling: `pip install quartodoc` (or, from the repo root,
  `pip install --group docs .`). The `dingo` package must be importable / on the path;
  the API build parses it statically, so its full runtime is not required.

## Building

From `source/`:

```sh
python gen_api.py     # refresh the API object list in _quarto.yml (the sphinx-apidoc analogue)
python build_api.py   # generate reference/*.qmd from docstrings (merges __init__ params)
quarto render         # build the site into _site/   (or: quarto preview)
```

The rendered site lands in `source/_site/` (git-ignored). Use `quarto preview` for a
live-reloading local preview.

Deployment to GitHub Pages is handled by `.github/workflows/docs.yml`.
