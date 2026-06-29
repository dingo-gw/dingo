---
name: build-docs
description: Build the dingo Sphinx documentation locally. Use when the user wants to build, preview, or check the docs after editing docstrings or files under docs/.
---

# Building the docs

Docs are Sphinx (RTD theme, myst-nb for notebooks, napoleon for NumPy docstrings).

```bash
cd docs && uv run make html
```

Output lands in `docs/build/html/` (open `index.html`). API pages are generated from
docstrings via `autodoc`; if you add new modules, RTD runs `sphinx-apidoc -o docs/source dingo`
(see `.readthedocs.yaml`) — run that locally too if new modules don't show up:

```bash
uv run sphinx-apidoc -o docs/source dingo && cd docs && uv run make html
```

Requires the dev extras (`uv sync` installs sphinx and the myst/mermaid/bibtex plugins).
