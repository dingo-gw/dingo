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

**Dingo** is a Python program for analyzing gravitational wave data using neural posterior estimation. It is based on the ideas presented in https://arxiv.org/abs/2106.12594.

.. note::
   This project is under active development.

.. toctree::
   :caption: Getting started
   :maxdepth: 2
   
   installation
   overview
   quickstart

.. toctree::
   :caption: User guide
   :maxdepth: 2

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
