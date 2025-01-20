# Inference

With a trained network, inference can be performed on injections or real data. For 
injections, see the [discussion in the examples](example_injection.md). For real data, we 
recommend to use [dingo_pipe](dingo_pipe.md).

## The `Sampler` class

Inference uses the `Sampler` class, or more specifically, the `GWSampler` class,
which inherits from it.

```{eval-rst}
.. autoclass:: dingo.gw.inference.gw_samplers.GWSampler
    :members:
    :inherited-members:
    :show-inheritance:
```

This is instantiated based on a `PosteriorModel`. To draw samples, the `context` property must first be set to the data to be analyzed. For gravitational waves this should be a dictionary with the following keys:

waveform
: (unwhitened) strain data in each detector

asds
: noise ASDs estimated in each detector at the time of the event

parameters (optional)
: for injections, the true parameters of the signal (for saving; ignored for sampling)

Once this is set, the `run_sampler()` method draws the requested samples from the posterior conditioned on the context. It applies some post-processing (to de-standardize the data, and to correct for the rotation of the Earth between the network reference time and the event time), and then stores the result as a DataFrame in `GWSampler.samples`. The DataFrame contains columns for each inference parameter, as well as the log probability of the sample under the posterior model.

The `GWSampler.metadata` attribute contains all settings that went into producing the samples, including training datasets, network training settings, event metadata (for real events) and possible injection parameters. Finally, the `to_samples_dataset()` method returns a `SamplesDataset` containing all results, including the samples, settings, and context. This can be saved easily as HDF5.


## Injections

Injections (i.e., simulated data) are produced using the `Injection` class. It includes options for fixed or random parameters (drawn from a prior), and it returns injections in a format that can be directly set as `GWSampler.context`.

```{eval-rst}
.. autoclass:: dingo.gw.injection.Injection
    :members:
    :show-inheritance:
```

```{hint}
The convenience class method `from_posterior_model_metadata()` instantiates an `Injection` with all of the settings that went into the posterior model. To this class pass the PosteriorModel.metadata dictionary. It should produce injections that perfectly match the characteristics of the training data (waveform approximant, data conditioning, noise characteristics, etc.). This can be very useful for testing a trained model.
```

```{important}
Repeated calls to `Injection.injection()`, even with the same parameters, will produce injections with different noise realizations (which therefore lead to different posteriors). For repeated analyses of the *exact same* injection (e.g., with different models or codes) it is necessary to either save the injection for re-use or fix a random seed.
```