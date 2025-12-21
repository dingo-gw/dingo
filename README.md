[![Python package](https://github.com/dingo-gw/dingo/actions/workflows/pytest.yml/badge.svg)](https://github.com/dingo-gw/dingo/actions/workflows/pytest.yml)
[![PyPI version](https://img.shields.io/pypi/v/dingo-gw.svg)](https://pypi.org/project/dingo-gw/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/dingo-gw)](https://anaconda.org/conda-forge/dingo-gw)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/dingo-gw)](https://anaconda.org/conda-forge/dingo-gw)



# Dingo

**Dingo (Deep Inference for Gravitational-wave Observations)** is a Python program for analyzing gravitational wave data using neural posterior
estimation. It dramatically speeds up inference of astrophysical source parameters from
data measured at gravitational-wave observatories. Dingo aims to enable the routine
use of the most advanced theoretical models in analyzing data, to make rapid predictions
for multi-messenger counterparts, and to do so in the context of sensitive detectors with
high event rates.

The basic approach of Dingo is to train a neural network to represent the Bayesian
posterior conditioned on data. This enables *amortized inference*: when new data are
observed, they can be plugged in and results obtained in a small amount of time. Tasks
handled by Dingo include

* building training datasets;
* training normalizing flows to estimate the posterior density;
* performing inference on real or simulated data; and
* verifying and correcting model results using importance sampling.

## Installation

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
To install using conda, first activate a conda environment, then run:

```sh
conda install -c conda-forge dingo-gw
```

### Development install

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

## Usage

For instructions on using Dingo, please refer to the [documentation](https://dingo-gw.readthedocs.io/en/latest/).

## References

Dingo is based on the following series of papers:

1. https://arxiv.org/abs/2002.07656: 5D toy model
2. https://arxiv.org/abs/2008.03312: 15D binary black hole inference
3. https://arxiv.org/abs/2106.12594: Amortized inference and group-equivariant neural posterior estimation
4. https://arxiv.org/abs/2111.13139: Group-equivariant neural posterior estimation
5. https://arxiv.org/abs/2210.05686: Importance sampling
6. https://arxiv.org/abs/2211.08801: Noise forecasting
7. https://arxiv.org/abs/2407.09602: Binary neutron star inference

Dingo was used also in https://arxiv.org/abs/2404.14286 to find evidence for eccentric binaries.

If you use Dingo in your work, we ask that you please cite at least
https://arxiv.org/abs/2106.12594.

Contributors to the code are listed in [AUTHORS.md](https://github.com/dingo-gw/dingo/blob/main/AUTHORS.md). We thank Charlie Hoy, Vivien Raymond,
and Rory Smith for acting as LIGO-Virgo-KAGRA (LVK) review chairs. Dingo makes use of
many LVK software tools, including [Bilby](https://lscsoft.docs.ligo.org/bilby/),
[bilby_pipe](https://lscsoft.docs.ligo.org/bilby_pipe/master/index.html), and
[LALSimulation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/), as well as third
party tools such as [PyTorch](https://pytorch.org) and
[nflows](https://github.com/bayesiains/nflows).

## Contact

For questions or comments please contact
[Maximilian Dax](mailto:maximilian.dax@tuebingen.mpg.de) or
[Stephen Green](mailto:stephen.green2@nottingham.ac.uk).
