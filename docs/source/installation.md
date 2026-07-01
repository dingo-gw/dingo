# Installation

## Standard

### Pip

To install using pip, run the following within a suitable virtual environment:
```sh
pip install dingo-gw
```
This installs Dingo and its runtime dependencies, as specified in  
[`pyproject.toml`](https://github.com/dingo-gw/dingo/blob/main/pyproject.toml).

Optional functionality can be enabled via extras, for example:

```sh
pip install "dingo-gw[wandb,pyseobnr]"
```

### Conda

Dingo is also available from the [conda-forge](https://conda-forge.org) repository.
To install using conda, first activate a conda environment, and then run
```sh
conda install -c conda-forge dingo-gw
```

## Development

### Installation

If you would like to make changes to Dingo or contribute to its development, install it
from source.

First clone the repository:

```sh
git clone git@github.com:dingo-gw/dingo.git
cd dingo-gw
```

#### Recommended (using `uv`)

We recommend using [`uv`](https://docs.astral.sh/uv/) for development installs, as it
provides fast, reproducible dependency resolution.

Create a virtual environment and install all development dependencies:

```sh
uv sync
```

This installs Dingo in editable mode along with development, documentation, and typing
dependencies. To also install optional dependencies, use

```sh
uv sync --extra wandb --extra pyseobnr
```

#### Alternative (pip)

If you prefer pip, create and activate a virtual environment:
```sh
python3 -m venv dingo-venv
source dingo-venv/bin/activate
```

Install Dingo in editable mode with development tools:
```sh
pip install -e ".[dev]"
```

Optional user-facing features can be enabled via extras, for example:
```sh
pip install -e ".[dev,wandb]"
```

### Documentation

To build the documentation, first generate the API documentation using [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html):
```sh
cd docs
sphinx-apidoc -o source ../dingo
```
This will create `dingo.*.rst` and `modules.rst` files in `source/`. These correspond to
the various modules and are constructed from docstrings.

To finally compile the documentation, run
```sh
make html
```
This creates a directory `build/` containing HTML documentation. The main index is at `build/html/index.html`.

To use the autodoc feature, which works for pycharm and numpy docstrings, insert in a .rst file, e.g.,

```
.. autofunction:: dingo.core.utils.trainutils.write_history`
```

This will render as 

```{eval-rst}
.. autofunction:: dingo.core.utils.trainutils.write_history
```

#### Cleanup

To remove generated docs, execute
```sh
make clean
rm source/dingo.* source/modules.rst
```
