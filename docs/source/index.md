<!-- dingo documentation master file, created by
   sphinx-quickstart on Thu Jan 13 14:37:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. If you want to build the docs do 
   `sphinx-build -b html docs/source/ docs/build/` from the dingo directory
   If this is causing and issue try `python3 -m sphinx.cmd.build -b html docs/source docs/build`
   to make sure you are using the correct python module. If you want to generate API-docs ie all of those 
   dingo.core.nn.rst files just run `sphinx-apidoc -o dingo/docs/source dingo/dingo` -->

Dingo
=====

**Dingo (Deep Inference for Gravitational-wave Observations)** is a Python program for analyzing gravitational wave data using neural posterior
estimation. It dramatically speeds up inference of astrophysical source parameters from
data measured at gravitational-wave observatories. Dingo aims to enable the routine
use of the most advanced theoretical models in analysing data, to make rapid predictions
for multi-messenger counterparts, and to do so in the context of sensitive detectors with
high event rates.

The basic approach of Dingo is to *train a neural network to represent the Bayesian
posterior*, conditioned on data. This enables **amortized inference**: when new data are
observed, they can be plugged in and results obtained in a small amount of time. Tasks
handled by Dingo include

* [building training datasets](waveform_dataset.ipynb);
* [training](training.md) normalizing flows to estimate the posterior density;
* [performing inference](inference.md) on real or simulated data; and
* verifying and correcting model results using [importance sampling](result.md#importance-sampling).

As training a network from scratch can be expensive, we intend to also distribute trained networks that can be used directly for inference. These can be used with [dingo_pipe](dingo_pipe.md) to automate analysis of gravitational wave events.

```{eval-rst}
.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   overview
   quickstart

.. toctree::
   :caption: Examples
   :maxdepth: 1
   
   example_toy_npe_model
   example_npe_model
   example_gnpe_model
   example_injection

.. toctree::
   :caption: Advanced guide
   :maxdepth: 1

   sbi
   code_design
   generating_waveforms
   waveform_dataset
   training_transforms
   noise_dataset
   network_architecture
   training
   inference
   gnpe
   result
   dingo_pipe
   
.. toctree::
   :caption: API
   :maxdepth: 1
   
   modules
```

References
----------

Dingo is based on a series of papers developing neural posterior estimation for gravitational waves, starting from proof of concept {cite:p}`Green:2020hst`, to inclusion of all 15 parameters and analysis of real data {cite:p}`Green:2020dnx`, noise conditioning and full amortization {cite:p}`Dax:2021tsq`, and group-equivariant NPE {cite:p}`Dax:2021myb`. Dingo results are augmented with importance sampling in {cite:p}`Dax:2022pxd`. Finally, training with forecasted noise (needed for training *prior* to an observing run) is described in {cite:p}`Wildberger:2022agw`.

```{eval-rst}
.. bibliography::
```

If you use Dingo in your work, we ask that you please cite at least {cite:p}`Dax:2021tsq`.

Contributors to the code are listed in [AUTHORS.md](https://github.com/dingo-gw/dingo/blob/main/AUTHORS.md). We thank Vivien Raymond
and Rory Smith for acting as LIGO-Virgo-KAGRA (LVK) code reviewers. Dingo makes use of
many LVK software tools, including [Bilby](https://lscsoft.docs.ligo.org/bilby/),
[bilby_pipe](https://lscsoft.docs.ligo.org/bilby_pipe/master/index.html), and
[LALSimulation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/), as well as third
party tools such as [PyTorch](https://pytorch.org) and
[nflows](https://github.com/bayesiains/nflows).

Contact
-------

For questions or comments please contact
[Maximilian Dax](mailto:maximilian.dax@tuebingen.mpg.de) or
[Stephen Green](mailto:stephen.green2@nottingham.ac.uk).

Indices and tables
------------------

```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```