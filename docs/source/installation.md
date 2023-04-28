# Installation

## Standard

### Pip

To install using pip, run the following within a suitable virtual environment:
```sh
pip install dingo-gw
```
This will install Dingo as well as all of its requirements, which are listed in
[pyproject.toml](https://github.com/dingo-gw/dingo/blob/main/pyproject.toml).

### Conda

Dingo is also available from the [conda-forge](https://conda-forge.org) repository.
To install using conda, first activate a conda environment, and then run
```sh
conda install -c conda-forge dingo-gw
```

## Development

If you would like to make changes to Dingo, or to contribute to its development, you
should install Dingo from source. To do so, first clone this repository:
```sh
git clone git@github.com:dingo-gw/dingo.git
```
Next create a virtual environment for Dingo, e.g.,
```sh
python3 -m venv dingo-venv
source dingo-venv/bin/activate
```
This creates and activates a [venv](https://docs.python.org/3/library/venv.html) for Dingo
called `dingo-venv`. In this virtual environment, install Dingo:
```sh
cd dingo
pip install -e ."[dev]"
```
This command installs an editable version of Dingo, meaning that any changes to the Dingo
source are reflected immediately in the installation. The inclusion of `dev` installs
extra packages needed for development (code formatting, compiling documentation, etc.)

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
