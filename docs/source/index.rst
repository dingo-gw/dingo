.. dingo documentation master file, created by
   sphinx-quickstart on Thu Jan 13 14:37:54 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. If you want to build the docs do 
   `sphinx-build -b html docs/source/ docs/build/` from the dingo-devel directory
   If this is causing and issue try `python3 -m sphinx.cmd.build -b html docs/source docs/build`
   to make sure you are using the correct python module. If you want to generate API-docs ie all of those 
   dingo.core.nn.rst files just run `sphinx-apidoc -o dingo-devel/docs/source dingo-devel/dingo`

Dingo
=====

**Dingo** is a Python program for analyzing gravitational wave data using neural posterior estimation. It contains code for

* building training datasets;
* training normalizing flows to estimate the posterior density;
* performing inference on real or simulated data; and
* verifying and correcting model results using importance sampling (**Dingo-IS**).

As training a network from scratch can be expensive, we intend to also distribute trained networks that can be used directly for inference.

.. note::
   This project is under active development.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   overview
   quickstart

.. toctree::
   :caption: Advanced guide
   :maxdepth: 1

   sbi
   design_philosophy
   generating_waveforms
   waveform_dataset
   training_transforms
   noise_dataset
   network_architecture
   training
   inference
   gnpe
   importance_sampling

References
----------

Dingo is based on a series of papers developing NPE for GW parameter inference, starting from proof of concept :cite:p:`Green:2020hst`, to inclusion of all 15 parameters and analysis of real data :cite:p:`Green:2020dnx`, noise conditioning and full amortization :cite:p:`Dax:2021tsq`, and group-equivariant NPE :cite:p:`Dax:2021myb`. Dingo results are augmented with importance sampling in :cite:p:`Dax:2022pxd`.

.. bibliography::

If you use Dingo, we ask that you cite at least :cite:p:`Dax:2021tsq`.

Dingo also makes use of several 3rd party packages, including

* `Bilby <https://lscsoft.docs.ligo.org/bilby/>`_
* `LALSimulation <https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/>`_
* `PyTorch <https://pytorch.org>`_
* `nflows <https://github.com/bayesiains/nflows>`_




Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
