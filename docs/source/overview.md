# Overview

Dingo performs gravitational-wave (GW) parameter estimation using [**neural posterior estimation**](sbi.md). The basic idea is to train a neural network (a normalizing flow) to represent the Bayesian posterior distribution $p(\theta|d)$ for GW parameters $\theta$ given observed data $d$. Training can take some time (typically, a week for a production-level model) but once trained, inference is very fast (just a few seconds). 

## Basic workflow

The basic workflow for using Dingo is as follows:

1. **Prepare training data.** This consists of pairs of intrinsic parameters and waveform polarizations, as well as noise PSDs. Training parameters are drawn from the prior distribution, and waveforms are simulated using a waveform model.
2. **Train a model.** Build a neural network and simulate data sets (noisy waveforms in detectors). Train the model to infer parameters based on the data.
3. **Perform inference on new data** using the trained model.

In many cases, a user may have downloaded a pre-trained model. If so, there is no need to carry out the first two steps, and instead skip to **step 3**.

## Command-line interface

In most cases, we expect Dingo to be called from the command line. Dingo commands begin with the prefix `dingo_`. There can be a large number of configurations options for many tasks, so in such cases, rather than specify all settings as arguments, Dingo commands take a single YAML file containing all settings. As described in the [](quickstart.md), it is best to begin with settings files provided in the examples/ folder, modifying them as necessary.

### Summary of commands

Here we provide a list of user commands along with brief descriptions. The commands for carrying out the main tasks above are

```{table}

| Command | Description |
|---|---|
|`dingo_generate_dataset`| Generate a training dataset of waveform polarizations. |
|`dingo_generate_ASD_dataset`| Generate a training dataset of detector noise ASDs. |
|`dingo_train`| Build and train a neural network. |
|`dingo_analyze_event`| Analyze GW data at a given time. |
```

Building a training dataset and training a model can be very expensive tasks. We therefore expect these to be frequently run on clusters, and for this reason provided HTCondor versions of these commands:

```{table}

| Command | Description |
|---|---|
|`dingo_generate_dataset_dag`| Condor version of `dingo_generate_dataset`. |
|`dingo_train_condor`| Condor version of `dingo_train`. |
```

Finally, there are several utility commands that are useful for working with Dingo-produced files:

```{table}

| Command | Description |
|---|---|
|`dingo_ls`| Inspect a file produced by Dingo and print a summary.|
|`dingo_append_training_stage`| Modify the training plan of a model checkpoint.|
```

```{hint}

The `dingo_ls` command is very useful for inspecting Dingo files. It will print all settings that wen in to producing the file, as well as some derived quantities.
```

### File types

As noted above, most Dingo commands take a YAML file to specify configuration options. When run, these commands generate data, which is usually stored in HDF5 files. One exception is when training a neural network. This saves the network weights using the PyTorch `.pt` format.

```{important}

In all cases, Dingo will save the YAML file settings within the final output file. This is needed for downstream tasks and for maintaining reproducibility.
```


## GNPE

A slightly more complicated workflow occurs when using [](gnpe.md). GNPE is an algorithm that combines physical symmetries with Gibbs sampling to significantly improve results. When using GNPE, however, it is necessary to train **two networks**---one main (conditional) network that will be repeatedly sampled during Gibbs sampling and one smaller network used to initialize the Gibbs sampler. At inference time, `dingo_analyze_event` must be pointed to **both** of these networks. See the section on GNPE [](gnpe.md#usage) for further details.