# Inference

With a trained network, inference can be performed on injections or real data. For 
injections, see the [discussion in the examples](example_injection.md). For real data, we 
recommend to use [dingo_pipe](dingo_pipe.md).

## The `GWComposedSampler` class

Inference uses the `GWComposedSampler` class, which represents the posterior as a
chain of conditional factors (the network flow, deterministic reparametrizations
such as the sky-frame rotation, and fillers for fixed parameters).

```{eval-rst}
.. autoclass:: dingo.gw.inference.sampler.GWComposedSampler
    :members:
    :inherited-members:
    :show-inheritance:
```

A plain-NPE sampler is built with `GWComposedSampler.from_model(model, event_data,
event_metadata)`, where `model` is a `PosteriorModel` and `event_data` is the data
to be analyzed---a dictionary with the following keys:

waveform
: (unwhitened) strain data in each detector

asds
: noise ASDs estimated in each detector at the time of the event

parameters (optional)
: for injections, the true parameters of the signal (for saving; ignored for sampling)

The `run_sampler()` method then draws the requested samples from the posterior conditioned on the data. All processing---de-standardization inside the network factor, and the correction for the rotation of the Earth between the network reference time and the event time---is expressed as chain steps, and the samples are stored as a DataFrame in `GWComposedSampler.samples`. The DataFrame contains columns for each inference parameter, as well as the log probability of the sample under the posterior model.

The `GWComposedSampler.metadata` attribute contains all settings that went into producing the samples. The `to_result()` method returns a [Result](dingo.gw.result.Result) containing the samples, settings (including structured sampler provenance under `settings["sampler"]`), and data; `to_hdf5()` saves it directly.


## Injections

Injections (i.e., simulated data) are produced using the `Injection` class. It includes options for fixed or random parameters (drawn from a prior), and it returns injections in a format that can be passed directly as the `event_data` of a `GWComposedSampler`.

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