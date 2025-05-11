# Toy Example

The goal of the following tutorial is to take a user from start to finish analyzing GW150914 using dingo.

```{caution}
This is only a toy example which is useful for testing on a local machine. This
is NOT meant be used for production gravitational wave analyses.
```

There are 4 main steps:

1. Generate the waveform dataset
2. Generate the ASD dataset
3. Train the network
4. Do inference

In this tutorial as well as the [npe model](example_npe_model) and [gnpe model](example_gnpe_model) the following file structure will
be employed

```
toy_npe_model/

    #  config files
    waveform_dataset_settings.yaml
    asd_dataset_settings.yaml
    train_settings.yaml
    GW150914.ini

    training_data/
        waveform_dataset.hdf5
        asd_dataset/ # Contains the asd_dataset.hdf5 and also temp files for asd generation

    training/
        model_050.pt
        model_stage_0.pt
        model_latest.pt
        history.txt
        #  etc...

    outdir_GW150914/
        #  dingo_pipe output
```

The config files which are the only ones which need to be edited are contained in the top level directory. In the next
few sections these config files will be explained. To download sample config files, please visit 
https://github.com/dingo-gw/dingo/tree/main/examples. In this tutorial the `toy_npe_model` folder will be used.


Step 1 Generating a waveform dataset
------------------------------------

After downloading the files for the tutorial first run

```
cd toy_npe_model/
mkdir training_data
mkdir training
```

to set up the file structure. Then run

```
dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5
```

which will create a 
{py:class}`dingo.gw.waveform_generator.waveform_generator.WaveformGenerator`
object and store it at the location provided with `--out_file`. For convenience, 
here is the waveform dataset file

```yaml
domain:
  type: UniformFrequencyDomain
  f_min: 20.0
  f_max: 1024.0
  delta_f: 0.25  # Expressions like 1.0/8.0 would require eval and are not supported

waveform_generator:
  approximant: IMRPhenomD
  f_ref: 20.0
  # f_start: 15.0  # Optional setting useful for EOB waveforms. Overrides f_min when generating waveforms.

# Dataset only samples over intrinsic parameters. Extrinsic parameters are chosen at train time.
intrinsic_prior:
  mass_1: bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)
  mass_2: bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)
  chirp_mass: bilby.gw.prior.UniformInComponentsChirpMass(minimum=15.0, maximum=100.0)
  mass_ratio: bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)
  phase: default
  chi_1: bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.9))
  chi_2: bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.9))
  theta_jn: default
  # Reference values for fixed (extrinsic) parameters. These are needed to generate a waveform.
  luminosity_distance: 100.0  # Mpc
  geocent_time: 0.0  # s

# Dataset size
num_samples: 10000

compression: None
```

The file `waveform_dataset_settings.yaml` contains four
sections: `domain`, `waveform_generator`, `intrinsic_prior`, and `compression`. The
domain section defines the settings for storing the waveform. Note the `type`
attribute; this does not refer to the native domain of the waveform model, but
rather to the internal {py:class}`dingo.gw.domains.Domain` class. This allows the use
of time domain waveform models, which are transformed into Fourier domain before
being passed to the network. Currently, only
the {py:class}`dingo.gw.domains.FrequencyDomain` class is supported for training the
network. It is sometimes advisable to generate waveforms with a higher `f_max` and then
truncate them at a lower `f_max` for training due to issues with generating short waveforms
for some of the waveform models implemented in LALSuite's LALSimulation package
(https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/).


The `waveform_generator` section specifies the `approximant` attribute.
At present any waveform model, aka `approximant`, that is callable through LALSimulation's
`SimInspiralFD` API can be used to generate waveforms for dingo via the
{py:class}`dingo.gw.waveform_generator.waveform_generator.WaveformGenerator` module (see
[generating_waveforms](generating_waveforms.md)).

The `intrinsic_prior` section is based on Bilby's prior module.
Default values can be found in `dingo.gw.prior`.
Two priors to note are the `chirp_mass` and `mass_ratio`, whose minimum values are set
to 15.0 and 0.125, respectively. Extending these priors towards lower chirp masses
or more extreme mass-ratios may lead to poor performance of the embedding network and normalizing 
flow during training and would require changes to the network setup.
Note that the `luminosity_distance` and `geocent_time` are defined as constants
to generate the waveform at a fixed reference point.

The compression section can be set to None for testing purposes. For a practical
example of how it is used, see the next tutorial.

Step 2 Generating the Amplitude Spectral Density (ASD) dataset
--------------------------------------------------------------

To generate an ASD dataset run

```
dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset
```

This command will generate an {py:class}`dingo.gw.noise.asd_dataset.ASDDataset` object in the form of an .hdf5 file, which will be used later for training. The reason for specifying a folder instead of a file, as in the waveform dataset example, is because some temporary data is downloaded to create Welch estimates of the ASD. This data can be removed later, but it is sometimes useful for understanding how the ASDs were estimated. For convenience here is a copy of the `asd_dataset_settings.yaml` file.

```yaml
dataset_settings:
f_s: 4096
time_psd: 1024
T: 4
window:
    roll_off: 0.4
    type: tukey
time_gap: 0          # specifies the time skipped between to consecutive PSD estimates. If set < 0, the time segments overlap
num_psds_max: 1  # if set > 0, only a subset of all available PSDs will be used
detectors:
    - H1
    - L1
observing_run: O1
```

The `asd_dataset_settings.yaml` file includes several attributes. `f_s` is the sampling frequency in Hz, `time_psd` is the length of time used for an ASD estimate, and `T` is the duration of each ASD segment. Thus, the value of `time_psd`/`T` gives the number of segments analyzed to estimate one ASD. To avoid spectral leakage, a window is applied to each segment. We use the standard window used in LVK analyses, a Tukey window with a roll off of $\alpha=0.4$. The next attribute, `num_psds_max=1`, defines the number of ASDs stored in the ASD dataset. For now, we will use only one. See the next [tutorial](example_npe_model.md) for a more advanced setup.

Step 3 Training the network
---------------------------

To train the network, first the paths to the correct datasets must be specified before executing:

```
dingo_train --settings_file train_settings.yaml --train_dir training
```

While this file contains numerous settings that are discussed in [training](training.md), we will cover the most significant ones here. Again here is the file.


```yaml
data:
  waveform_dataset_path: training_data/waveform_dataset.hdf5  # Contains intrinsic waveforms
  train_fraction: 0.95
  window:  # Needed to calculate window factor for simulated data
    type: tukey
    f_s: 4096
    T: 4.0
    roll_off: 0.4
  detectors:
    - H1
    - L1
  extrinsic_prior:  # Sampled at train time
    dec: default
    ra: default
    geocent_time: bilby.core.prior.Uniform(minimum=-0.10, maximum=0.10)
    psi: default
    luminosity_distance: bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)
  ref_time: 1126259462.391
  inference_parameters: 
  - chirp_mass
  - mass_ratio
  - chi_1
  - chi_2
  - theta_jn
  - dec
  - ra
  - geocent_time
  - luminosity_distance
  - psi
  - phase

# Model architecture
model:
  posterior_model_type: normalizing_flow
  # kwargs for neural spline flow
  posterior_kwargs:
    num_flow_steps: 5
    base_transform_kwargs:
      hidden_dim: 64 
      num_transform_blocks: 5
      activation: elu
      dropout_probability: 0.0
      batch_norm: True
      num_bins: 8
      base_transform_type: rq-coupling
  # kwargs for embedding net
  embedding_kwargs:
    output_dim: 128
    hidden_dims: [1024, 512, 256, 128]
    activation: elu
    dropout: 0.0
    batch_norm: True
    svd:
      num_training_samples: 1000
      num_validation_samples: 100
      size: 50

# The first stage (and only) stage of training. 
training:
  stage_0:
    epochs: 20
    asd_dataset_path: training_data/asd_dataset/asds_O1.hdf5  # this should just contain a single fiducial ASD per detector for pretraining
    freeze_rb_layer: True
    optimizer:
      type: adam
      lr: 0.0001
    scheduler:
      type: cosine
      T_max: 20
    batch_size: 64

# Local settings for training that have no impact on the final trained network.
local:
  device: cpu  # Change this to 'cuda' for training on a GPU.
  num_workers: 6  # num_workers >0 does not work on Mac, see https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
  runtime_limits:
    max_time_per_run: 36000
    max_epochs_per_run: 30
  checkpoint_epochs: 15
  leave_waveforms_on_disk: True
```

For training, several `extrinsic_priors` are set, which project the waveforms generated in step 1 onto the detector network according to the specified priors. This is considerably cheaper than generating waveforms sampled from the full intrinsic plus extrinsic prior in step 1.

Another crucial setting is `inference_parameters`. By default, all the parameters described in `dingo.gw.prior` are inferred. If a parameter needs to be marginalized over, this parameter can be omitted from `inference_parameters`.

Essential settings for the model architecture of the normalizing flow (i.e., the neural spline flow and the embedding network) are as follows: `posterior_kwargs.num_flow_steps` describes the number of flow transforms from the base distribution to the final distribution, while `embedding_net_kwargs.hidden_dim` defines the dimensions of the neural network's hidden layer, which selects the most important data features. Finally, `embedding_net_kwargs.svd` describes the settings of the SVD used as a pre-processing step before passing data vectors to the embedding network. For a production network, these values should be much higher than those used in this tutorial.

Next, we turn to the training section. Here we only employ a single stage of training with settings provided under the `stage_0` attribute. This stage uses the training dataset generated in step 1 for 30 epochs. We also specify the `asd_dataset_path` here, which was created in step 2.

Finally, the local settings section specifies technical details of the training setup. It contains information about, e.g., parallelization during training and the device used. An important setting here is `num_workers`, which determines how many PyTorch dataloader processes are spawned during training. If training is too slow, a potential cause is a lack of workers to load data into the network. This can be identified if the dataloader times in the `dingo_train` output exceed 100ms. The solution is generally to increase the number of workers.

Step 4 Doing Inference
----------------------

The final step is to do inference, for example on GW150914. To do this we will use
[dingo_pipe](dingo_pipe.md). For a local run execute:

```
dingo_pipe GW150914.ini
```

This calls `dingo_pipe` on an INI file that specifies the event to run on,
```ini
################################################################################
##  Job submission arguments
################################################################################

local = True
accounting = dingo
request-cpus-importance-sampling = 2

################################################################################
##  Sampler arguments
################################################################################

model = training/model_latest.pt
device = 'cpu'
num-samples = 5000
batch-size = 5000
recover-log-prob = false
importance-sample = false

################################################################################
## Data generation arguments
################################################################################

trigger-time = GW150914
label = GW150914
outdir = outdir_GW150914
channel-dict = {H1:GWOSC, L1:GWOSC}
psd-length = 128
# sampling-frequency = 2048.0
# importance-sampling-updates = {'duration': 4.0}

################################################################################
## Plotting arguments
################################################################################

plot-corner = true
plot-weights = true
plot-log-probs = true
```

This will generate files which are described in [dingo_pipe](dingo_pipe.md). To see the results, take a look in `outdir_GW150914`. We set the flag `importance-sample = False` in the INI file, which disables importance sampling for this simple example. Generally one would omit this (it defaults to True).

We can load and manipulate the data with the following code. For example, here we create a cornerplot

```
from dingo.gw.result import Result
result = Result(file_name="outdir_GW150914/result/GW150914_data0_1126259462-4_sampling.hdf5")
result.plot_corner()
```

Notice the results don't look very promising, but this is expected as the settings used in this
example are not enough to warrant convergence. Dingo should also automatically generate a cornerplot which will
be displayed under outdir_GW150914.
