# dingo_pipe

Dingo includes a command-line tool **dingo_pipe** for automating inference tasks. This is based *very closely* on the [bilby_pipe](https://lscsoft.docs.ligo.org/bilby_pipe/master/index.html) package, with suitable modifications. The basic usage is to pass a `.ini` file containing event information and run configuration settings, e.g.,
```bash
dingo_pipe GW150914.ini
```
dingo_pipe then executes various commands for [preparing data](#data-generation), [sampling from networks](#sampling), [importance sampling](#importance-sampling), and [plotting](#plotting). It can execute commands locally or on a cluster using a DAG. This documentation will only describe the relevant differences compared to bilby_pipe, and we refer the reader to the bilby_pipe documentation for additional information.

```{code-block} ini
---
caption: Example `GW150914.ini` file. This is also available in the examples/ directory.
---
################################################################################
##  Job submission arguments
################################################################################

local = True
accounting = dingo
request-cpus-importance-sampling = 16
n-parallel = 4
sampling-requirements = [TARGET.CUDAGlobalMemoryMb>40000]
extra-lines = [+WantsGPUNode = True]
simple-submission = false

################################################################################
##  Sampler arguments
################################################################################

model-init = /data/sgreen/dingo-experiments/XPHM/O1_init/model_stage_1.pt
model = /data/sgreen/dingo-experiments/XPHM/testing_inference/model.pt
device = 'cuda'
num-gnpe-iterations = 30
num-samples = 50000
batch-size = 50000
recover-log-prob = true
importance-sample = true
prior-dict = {
luminosity_distance = bilby.gw.prior.UniformComovingVolume(minimum=100, maximum=2000, name='luminosity_distance'),
}

################################################################################
## Data generation arguments
################################################################################

trigger-time = GW150914
label = GW150914
outdir = outdir_GW150914
channel-dict = {H1:GWOSC, L1:GWOSC}
psd-length = 128
sampling-frequency = 2048.0
importance-sampling-updates = {'duration': 4.0}

################################################################################
## Calibration marginalization arguments
################################################################################

calibration-model = CubicSpline
spline-calibration-envelope-dict = {H1: GWTC1_GW150914_H_CalEnv.txt, L1: GWTC1_GW150914_L_CalEnv.txt}
spline-calibration-nodes = 10
spline-calibration-curves = 1000

################################################################################
## Plotting arguments
################################################################################

plot-corner = true
plot-weights = true
plot-log-probs = true
```

The main difference compared to a bilby_pipe `.ini` file is that one specifies trained Dingo models rather than data conditioning and prior settings. The reason for this is that such settings have already been incorporated into training of the model. It is therefore not possible to change them when sampling from the Dingo model. Understandably, this could cause inconvenience if one is interested in a different prior or data conditioning settings. As a solution, Dingo enables the changing of such settings during importance sampling, which applies the new settings for likelihood evaluations.

```{important}
For dingo_pipe it is necessary to specify a trained Dingo model *instead* of sampler
settings such as prior and data conditioning.
```

## Data generation

The first step is to download and prepare gravitational-wave data. In the example, dingo_pipe (using bilby_pipe routines) downloads the event and PSD data at the time of GW150914. It then prepares the data based on conditioning settings in the specified Dingo model. If other conflicting conditioning settings are provided (e.g., `sampling_frequency = 2048.0`), dingo_pipe stores these in the dictionary `importance_sampling_updates` (which can also be specified explicitly). These settings are ignored for now, and only applied later for calculating the likelihood in importance sampling.

The prepared event data and ASD are stored in a {py:class}`dingo.gw.data.event_dataset.EventDataset`, which is then saved to disk in HDF5 format.

```{note}
Dingo models are typically trained using Welch PSDs. For this reason we do not recommend using a BayesWave PSD for initial sampling. Rather, a BayesWave PSD should be specified within the `importance_sampling_updates` dictionary, so that it will be used during importance sampling.
```

## Sampling

The next step is sampling from the Dingo model. The model is loaded into a [GWSampler](dingo.gw.inference.gw_samplers.GWSampler) or [GWSamplerGNPE](dingo.gw.inference.gw_samplers.GWSamplerGNPE) object. (If using [GNPE](gnpe) it is necessary to specify a `model-init`.) The Sampler `context` is then set from the EventDataset prepared in the previous step. `num-samples` samples are then generated in batches of size `batch-size`. The samples (and context) are stored in a [Result](dingo.gw.result.Result) object and saved in HDF5 format.

If using GNPE, one can optionally specify `num-gnpe-iterations` (it defaults to 30). Importantly, obtaining the log probability when using GNPE requires an [extra step of training an unconditional flow](result.md#density-recovery). This is done using the `recover-log-prob` flag, which defaults to `True`. The default density recovery settings can be overwritten by providing a `density-recovery-settings` dictionary in the `.ini` file.

Since sampling uses GPU hardware, there is an additional key `sampling-requirements` for HTCondor requirements during the sampling stage. This is intended for specifying GPU requirements such as memory or CUDA version.

## Importance sampling

For importance sampling, the Result saved in the previous step is loaded. Since this contains the strain data and ASDs, as well as all settings used for training the network, the likelihood and prior can be evaluated for each sample point. If it is necessary to change data conditioning or PSD for importance sampling (i.e., if the `importance-sampling-updates` dictionary is non-empty), then a second [data generation](#data-generation) step is first carried using the new settings, and used as importance sampling context. The importance sampled result is finally saved as HDF5, including the estimated Bayesian evidence.

If a `prior-dict` is specified in the `.ini` file, then this will be used for the importance sampling prior. One example where this is useful is for the luminosity distance prior. Indeed, Dingo tends to train better using a uniform prior over luminosity distance, but physically one would prefer a uniform in volume prior. By specifying a `prior-dict` this change can be made in importance sampling.

```{caution}
If extending the prior support during importance sampling, be sure that the posterior does not rail up against the prior boundary being extended.
```

By default, dingo_pipe assumes that it is necessary to sample the phase synthetically, so it will do so before importance sampling. This can be turned off by passing an empty dictionary to `importance-sampling-settings`. Note that importance sampling itself can be switched off by setting the `importance-sample` flag to False (it defaults to True). 

Importance sampling (including synthetic phase sampling) is an expensive step, so dingo_pipe allows for parallelization: this step is split over `n-parallel` jobs, each of which uses `request-cpus-importance-sampling` processes. In the backend, this makes use of the Result [split()](dingo.core.result.Result.split) and [merge()](dingo.core.result.Result.merge) methods.

### Calibration marginalization

Settings related to calibration are used to **marginalize** over calibration uncertainty during importance sampling.

calibration-model
: None or "CubicSpline". If "CubicSpline", perform calibration marginalization using a cubic spline calibration model. If None do not perform calibration marginalization. (Default: None)

spline-calibration-envelope-dict
: Dictionary pointing to the spline calibration envelope files. This is required if calibration-model is "CubicSpline".

spline-calibration-nodes
: Number of calibration nodes. (Default: 10)

spline-calibration-curves
: Number of calibration curves to use for marginalization. (Default: 1000)

## Plotting

The standard Result [plots](result.md#plotting) are turned on using the `plot-corner`, `plot-weights`, and `plot-log-probs` flags.

## Additional options

extra-lines
: Additional lines for all submission scripts. This could be useful for particular cluster configurations.

simple-submission
: Strip the keys `accounting_tag`, `getenv`, `priority`, and `universe` from submission scripts. Again useful for particular cluster configurations.
