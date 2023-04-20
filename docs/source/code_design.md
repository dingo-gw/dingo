# Code design

## Reproducibility

Generating reproducible results must be central to any deep learning code. Dingo attempts to achieve this in the following ways:

### Settings

There are a large number of configuration options that must be selected when using Dingo. These include
* Waveform and noise dataset settings,
* Training settings, including pre-processing, neural network, and training strategy settings,
* Inference settings, including event time or injection data.

The Dingo approach is to save all of these settings as nested dictionaries together with the outputs of the various tasks. In practice, this means specifying the settings as a `.yaml` file and passing this to a command-line script that runs some code and produces an output file (`.hdf5` or `.pt`). The output file then contains the settings dictionary (possibly augmented by additional derived parameters). All output files can be inspected using the command-line script `dingo_ls`, which prints the stored settings and possibly additional information. The output from `dingo_ls` could (with a small amount of effort) be used to reproduce the exact results **(modulo random seeds, to be implemented)**.

In addition to saving the user-provided settings at each step, Dingo also saves the settings from precursor steps. For example, when training a model on data from a given waveform dataset, the waveform dataset settings are also saved along with the model settings. This can be very useful at a later point, when only the trained model is available, not the training data. Beyond ensuring reproducibility, having these precursor settings available is needed for certain downstream tasks (e.g., combining the intrinsic prior from a waveform dataset with the extrinsic prior specified for training).

### Random seeds

```{admonition}  To-do
Implement this.
```

### Unique identifiers for models

```{admonition}  To-do
Implement this.
```

## Code re-use

### `core` and `gw` packages

Although the only current use case for Dingo is to analyze LVK data, we hope that it can be extended to other GW or astrophysical (or more general scientific) applications. To facilitate this, we follow the [Bilby](https://lscsoft.docs.ligo.org/bilby/index.html) approach of partitioning code into `core` and `gw` components: `gw` contains GW-specific code (relating to waveforms, interferometers, etc.) whereas `core` contains generic network architectures, data structures, samplers, etc., that we expect could be used in other applications. As we find ways to write elements of code in more generic ways, we hope to migrate additional components from `gw` to `core`. We could then envision future packages, e.g., for LISA inference, GW populations, or cosmology.

### Data transforms

We follow the [PyTorch guidelines](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) of pre-processing data using a sequence of transforms. Dingo includes [transforms](training_transforms.ipynb) for tasks such as sampling extrinsic parameters, projecting waveform polarizations to detectors, and adding noise. The same transforms are re-used at inference time, where a similar (but always identical) sequence is required. Some transforms also behave differently at inference time, and thus have a flag to specify the mode.

### Data structures

Dingo uses several dataset classes, all of which inherit from {py:class}`dingo.core.dataset.DingoDataset`. This provides a common IO (to save/load from HDF5 as well as dictionaries). It also stores the settings dictionary as an attribute.

## Command-line scripts

In general, Dingo is constructed around libraries and classes that are used to carry out various data processing tasks. There are a large number of configuration options, which are often passed as dictionaries, enabling the addition of new settings without breaking old code.

For very high-level tasks, such as generating a training dataset or training a network, we believe it is most straightforward to use a command-line interface. This is because these are end-user tasks that might be called by separate programs, or on a cluster, or  because some of these (dataset generation and training) can be quite expensive.

A Dingo command-line script begins with the prefix `dingo_` and is usually a thin wrapper around a function that could be called by other code if desired. It takes as input a `.yaml` file, passes it as a dictionary to the function, obtains a result, and saves it to disk. We hope that this balance between libraries and a command-line interface enables an extensible code going forward.
