# Inference

With a trained network, inference means drawing posterior samples for a specific
stretch of data. There are two routes. For real events we recommend
[dingo_pipe](dingo_pipe.md), which downloads and prepares the data, runs the sampler,
and importance samples in one workflow. This page describes the underlying Python
interface, which is useful for injections, custom data, and interactive work.

## Event data

A sampler analyzes an `event_data` dictionary with the following entries:

waveform
: (unwhitened) strain data for each detector

asds
: noise amplitude spectral densities estimated for each detector at the time of the
  event

parameters (optional)
: for injections, the true parameters of the signal (stored with the results; ignored
  for sampling)

The data must be consistent with the model: the same detectors, and frequency content
covering the model's data domain. Data prepared by dingo_pipe is stored as an
{py:class}`~dingo.gw.data.event_dataset.EventDataset`, whose `data` attribute has
exactly this form, and the [Injection](#injections) class below produces it directly.

## Building and running a sampler

Inference uses the `GWComposedSampler` class, which represents the posterior as a
[chain of steps](sampling_chains.md). It is built from a trained model and the event
data:

```python
from dingo.core.posterior_models import build_model_from_kwargs
from dingo.gw.inference.sampler import GWComposedSampler

model = build_model_from_kwargs(
    filename="/path/to/model.pt", device="cuda", load_training_info=False
)
sampler = GWComposedSampler.from_model(model, event_data, event_metadata)
sampler.run_sampler(num_samples=50_000, batch_size=10_000)
```

The constructors cover the standard analysis types:

`from_model`
: A single-network model: plain NPE, or a prior-conditioned model with
  `fixed_context_parameters` (see [binary neutron stars](bns.md)).

`from_gnpe_models`
: Multi-iteration [GNPE](gnpe.md), from an initialization and a main model.

`from_singlestep_gnpe`
: Single-step GNPE with an explicit proxy source, used for
  [density recovery](result.md#density-recovery).

`run_sampler()` draws the requested number of samples in batches of `batch_size` and
stores them as a DataFrame in `sampler.samples`, with one column per inference
parameter plus `log_prob`, the log density of each sample under the model (absent for
density-free GNPE chains). All processing, including the de-standardization of
network outputs and the rotation of the right ascension from the training reference
frame to the event frame, is expressed as chain steps.

The `metadata` attribute contains all settings that went into producing the samples.
`to_result()` exports a [Result](result.md) containing the samples, the settings
(including the structured sampler provenance under `settings["sampler"]`), and the
event data; `to_hdf5()` saves it directly. Importance sampling and plotting then
proceed on the `Result`.

```{eval-rst}
.. autoclass:: dingo.gw.inference.sampler.GWComposedSampler
    :members:
    :inherited-members:
    :show-inheritance:
```

## Injections

Injections (simulated signals in stationary Gaussian noise) are produced with the
`Injection` class. It supports fixed parameters (`injection(theta)`) and random
parameters drawn from a prior (`random_injection()`), and it returns data in the
`event_data` format above, ready to pass to a sampler.

```{eval-rst}
.. autoclass:: dingo.gw.injection.Injection
    :members:
    :show-inheritance:
```

```{hint}
The class method `from_posterior_model_metadata()` instantiates an `Injection` with
the settings of a trained model (waveform approximant, data conditioning, detectors,
priors), so that its injections match the characteristics of the training data. This
is very useful for testing a model; see the [worked example](example_injection.md).
```

```{important}
Repeated calls to `Injection.injection()`, even with the same parameters, produce
different noise realizations and therefore different posteriors. For repeated
analyses of the exact same injection (e.g., with different models or codes), either
save the injection for re-use or fix a random seed.
```
