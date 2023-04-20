# Overview

Dingo performs gravitational-wave (GW) parameter estimation using [**neural posterior estimation**](sbi.md). The basic idea is to train a neural network (a normalizing flow) to represent the Bayesian posterior distribution $p(\theta|d)$ for GW parameters $\theta$ given observed data $d$. Training can take some time (typically, a week for a production-level model) but once trained, inference is very fast (just a few seconds). 

## Basic workflow

The basic workflow for using Dingo is as follows:

1. **Prepare training data.** This consists of pairs of intrinsic parameters and [waveform polarizations](waveform_dataset.ipynb), as well as [noise PSDs](noise_dataset.ipynb). Training parameters are drawn from the prior distribution, and [waveforms are simulated](generating_waveforms.ipynb) using a waveform model.
2. **Train a model.** [Build a neural network](network_architecture.ipynb) and [simulate data sets](training_transforms.ipynb) (noisy waveforms in detectors). [Train the model](training.md) to infer parameters based on the data.
3. **[Perform inference](dingo_pipe.md) on new data** using the trained model.

In many cases, a user may have downloaded a pre-trained model. If so, there is no need to carry out the first two steps, and one may instead skip to **step 3**.

## Command-line interface

In most cases, we expect Dingo to be called from the command line. Dingo commands begin with the prefix `dingo_`. There can be a large number of configurations options for many tasks, so in such cases, rather than specify all settings as arguments, Dingo commands take a single YAML or INI file containing all settings. As described in the [quickstart tutorial](quickstart.md), it is best to begin with settings files provided in the [examples/](https://github.com/dingo-gw/dingo/tree/main/examples) folder, modifying them as necessary.

### Summary of commands

Here we provide a list of key user commands along with brief descriptions. The commands for carrying out the main tasks above are

```{table}

| Command | Description |
|---|---|
|`dingo_generate_dataset`| Generate a training dataset of waveform polarizations. |
|`dingo_generate_ASD_dataset`| Generate a training dataset of detector noise ASDs. |
|`dingo_train`| Build and train a neural network. |
|`dingo_pipe`| Perform inference on data (real or simulated), starting from an INI file. |
```

Building a training dataset and training a model can be very expensive tasks. We therefore expect these to be frequently run on clusters, and for this reason provided [HTCondor](https://htcondor.readthedocs.io/en/latest/) versions of these commands (note that `dingo_pipe` is already HTCondor-compatible):

```{table}

| Command | Description |
|---|---|
|`dingo_generate_dataset_dag`| HTCondor version of `dingo_generate_dataset`. |
|`dingo_train_condor`| HTCondor version of `dingo_train`. |
```

Finally, there are several utility commands that are useful for working with Dingo-produced files:

```{table}

| Command | Description |
|---|---|
|`dingo_ls`| Inspect a file produced by Dingo and print a summary.|
|`dingo_append_training_stage`| Modify the training plan of a model checkpoint.|
|`dingo_pt_to_hdf5`| Convert a trained Dingo model from a PyTorch pickle .pt file to HDF5.|
```

```{hint}

The `dingo_ls` command is very useful for inspecting Dingo files. It will print all settings that went in to producing the file, as well as some derived quantities.
```

### File types

As noted above, most Dingo commands take a YAML file to specify configuration options (except for `dingo_pipe`, which uses an INI file, as is standard for LVK parameter estimation). When run, these commands generate data, which is usually stored in HDF5 files. One exception is when training a neural network. This saves the network weights using the PyTorch `.pt` format. However, primarily for LVK use, `dingo_pt_to_hdf5` can convert the weights of a trained model to a HDF5 file.

```{important}

In all cases, Dingo will save the YAML file settings within the final output file. This is needed for downstream tasks and for maintaining reproducibility.
```


## GNPE

A slightly more complicated workflow occurs when using [](gnpe.md). GNPE is an algorithm that combines physical symmetries with Gibbs sampling to significantly improve results. When using GNPE, however, it is necessary to train **two networks**---one main (conditional) network that will be repeatedly sampled during Gibbs sampling and one smaller network used to initialize the Gibbs sampler. At inference time, `dingo_pipe` must be pointed to **both** of these networks. See the section on [GNPE usage](gnpe.md#usage) for further details.
