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
This will install Dingo as well as all of its requirements, which are listed in
[pyproject.toml](https://github.com/dingo-gw/dingo/blob/main/pyproject.toml).

### Conda

Dingo is also available from the [conda-forge](https://conda-forge.org) repository.
To install using conda, first activate a conda environment, and then run
```sh
conda install -c conda-forge dingo-gw
```

### Development install

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

If you use Dingo in your work, we ask that you please cite at least
https://arxiv.org/abs/2106.12594.

Contributors to the code are listed in [AUTHORS.md](https://github.com/dingo-gw/dingo/blob/main/AUTHORS.md). We thank Vivien Raymond
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
