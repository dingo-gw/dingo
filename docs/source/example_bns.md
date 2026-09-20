# DINGO-BNS (binary neutron stars)

This example trains and runs a chirp-mass-conditioned network for binary neutron stars
with the [DINGO-BNS](bns.md) method. The standard tools are used throughout, with three
additions: the waveform dataset is heterodyned before compression, the multibanded
domain is determined from heterodyned waveforms, and the training transforms apply
chirp-mass GNPE. The configuration in `examples/binary_neutron_stars` follows the
GW170817 network of {footcite:p}`Dax:2024mcn`: 128 s segments ($\delta f = 1/128$ Hz),
an analysis band from 23 Hz to 1536 Hz, a low-spin prior ($a \leq 0.05$), 30 million
waveforms, and a kernel of half-width $0.005\,M_\odot$ on the chirp mass. It departs
from the paper in one respect: this network infers the sky position, whereas the
GW170817 network of the paper conditioned on it (pinned `ra` and `dec`).

## File structure

```
binary_neutron_stars/

    #  config files
    waveform_dataset_settings_ufd.yaml
    waveform_dataset_settings_mfd.yaml  # written by dingo_generate_multibanded_domain
    asd_dataset_settings.yaml
    asd_dataset_settings_fiducial.yaml
    train_settings.yaml
    GW170817.ini

    training_data/
        waveform_dataset.hdf5
        asd_dataset_fiducial/
        asd_dataset/

    training/
        model_latest.pt
        model_stage_0.pt
        history.txt
        ...

    outdir_GW170817/
        #  dingo_pipe output
```

## Step 1: Generate a waveform dataset

The dataset settings are first written for the uniform base domain, in
`waveform_dataset_settings_ufd.yaml`: a `UniformFrequencyDomain` from 20 Hz to 2048 Hz
with `delta_f: 0.0078125` (128 s segments); the tidal approximant
`IMRPhenomXP_NRTidalv3` with `spin_conversion_phase: 0.0`, which a phase-marginalized
network requires; and an `intrinsic_prior` with BNS mass and spin ranges (the `default`
entries are tuned to BBH) and priors on the tidal deformabilities, `lambda_1: default`
(`Uniform(0, 5000)`) or an explicit prior string as for `lambda_2`. The `compression`
block requests phase heterodyning:

```yaml
compression:
  whitening: aLIGO_ZERO_DET_high_P_asd.txt
  phase_heterodyning:
    order: 0
  svd:
    size: 200
    num_training_samples: 50000
    num_validation_samples: 10000
```

Each waveform is heterodyned at its own chirp mass before the SVD, so the basis can
compress the long inspirals; `order: 0` removes the leading-order chirp phase of
[phase heterodyning](bns.md#phase-heterodyning). The heterodyne belongs to the internal
storage only: the dataset inverts it on decompression, and the training transforms
re-heterodyne at the proxy (Step 3). Since heterodyning must precede decimation, a
multibanded dataset requires an approximant with a frequency-domain implementation
(see [waveform dataset](waveform_dataset.ipynb)).

The multibanded domain is then determined from these settings with the
[band tool](waveform_dataset.ipynb#generating-a-multibanded-domain):

```bash
dingo_generate_multibanded_domain --settings_file waveform_dataset_settings_ufd.yaml \
    --target_median_mismatch 1e-5 --chirp_mass_proxy_offset 0.005 --num_processes N
```

This writes `waveform_dataset_settings_mfd.yaml`, the same settings with a
`MultibandedFrequencyDomain`. The offset is the half-width of the training kernel, so
that the bands hold for data heterodyned anywhere in the kernel (see
[multibanding heterodyned data](bns.md#multibanding-heterodyned-data));
`dingo_evaluate_multibanded_domain` accepts the same option. With the settings above,
the target of $10^{-5}$ reproduces the banding of the network of
{footcite:p}`Dax:2024mcn` (eight bands, about 3700 bins from 20 Hz to 2048 Hz). The
dataset is generated from the multibanded settings file as usual:

```bash
dingo_generate_dataset --settings_file waveform_dataset_settings_mfd.yaml \
    --num_processes N --out_file training_data/waveform_dataset.hdf5
```

```{warning}
The NRTidalv3 approximants of LALSimulation (`IMRPhenomXP_NRTidalv3`,
`IMRPhenomXAS_NRTidalv3`) place the time origin of the waveform at the merger
frequency or, if the requested frequencies end below it, at the last requested
frequency. Training waveforms are evaluated at the multibanded frequencies, whereas
injections and likelihood templates are generated on the uniform dataset domain (and
then restricted to the analysis range), so the dataset domain must extend past the
merger frequency of the whole prior, as the 2048 Hz of the example does; a dataset
domain ending below the merger would shift the training waveforms in time relative to
the data. A `domain_update` or a frequency update at inference does not change the
domain templates are generated on. `IMRPhenomPv2_NRTidal` does not have this dependence.
```

## Step 2: Generate the noise datasets

The ASD datasets are generated as for any other network (see
[noise dataset](noise_dataset.ipynb)), on the uniform base domain with 128 s segments
(`T: 128.0` in `asd_dataset_settings.yaml`): a fiducial dataset with a single ASD per
detector for pre-training and a full dataset for fine-tuning, at the paths named in
`train_settings.yaml`. They are decimated to the multibanded domain automatically when
training starts.

```bash
dingo_generate_asd_dataset --settings_file asd_dataset_settings_fiducial.yaml \
    --data_dir training_data/asd_dataset_fiducial \
    --out_name training_data/asd_dataset_fiducial/asds_O2_fiducial.hdf5
dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml \
    --data_dir training_data/asd_dataset \
    --out_name training_data/asd_dataset/asds_O2.hdf5
```

The two settings files differ only in `num_psds_max` (1 for the fiducial dataset); see
the [NPE example](example_npe_model.md) for the two-stage arrangement.

## Step 3: Train the network

```bash
dingo_train --settings_file train_settings.yaml --train_dir training
```

The `data` section of `train_settings.yaml` configures chirp-mass GNPE and the
inference parameters:

```yaml
data:
  waveform_dataset_path: training_data/waveform_dataset.hdf5
  domain_update:
    f_min: 23.0
    f_max: 1536.0
  gnpe_chirp:
    kernel:
      chirp_mass: bilby.core.prior.Uniform(minimum=-0.005, maximum=0.005)
    order: 0
  inference_parameters:
    - delta_chirp_mass
    - mass_ratio
    # ... the remaining parameters, without chirp_mass or phase
    - lambda_1
    - lambda_2
```

`gnpe_chirp` inserts the chirp-mass GNPE transform after the extrinsic parameters are
sampled. The chirp mass is blurred by the kernel, a Bilby prior on the offset, to give
`chirp_mass_proxy`; the polarizations are heterodyned at the proxy, and the network
conditions on it (`chirp_mass_proxy` is appended to `context_parameters`
automatically). The offset `delta_chirp_mass` is available as an inference parameter
and takes the place of `chirp_mass` in `inference_parameters`; the chain restores
`chirp_mass` at inference (see [prior conditioning](bns.md#prior-conditioning)). The
SVD that seeds the embedding network is built from heterodyned waveforms.

`inference_parameters` is listed explicitly, including `lambda_1` and `lambda_2` (the
`default` list contains only the 15 BBH parameters), and without `phase`: the network
is phase marginalized, and the phase is reconstructed synthetically before importance
sampling, which is why the dataset sets `spin_conversion_phase: 0.0`. Additional
context parameters can be listed under `context_parameters`, and a parameter listed
there is left out of `inference_parameters` and pinned to a value per event at
inference (Step 4); the GW170817 network of {footcite:p}`Dax:2024mcn` conditioned on
the sky position in this way, and this configuration infers it instead. The remaining
settings (model, training stages, local) follow the standard [training](training.md)
layout, and training runs with `dingo_train` or `dingo_train_condor`.

## Step 4: Inference

```bash
dingo_pipe GW170817.ini
```

The chirp-mass proxy, this network's only context parameter, is pinned per event with
`fixed-context-parameters`, for instance to the value of a search trigger, or
determined from the data with `chirp-mass-scan` (see the
[chirp-mass scan](bns.md#the-chirp-mass-scan)). Both options are described under
[sampling](dingo_pipe.md#sampling) on the dingo_pipe page.

```{code-block} ini
---
caption: Sampler and data sections of a GW170817 configuration.
---
################################################################################
##  Sampler arguments
################################################################################

model = training/model_latest.pt
device = 'cuda'
num-samples = 50000
batch-size = 50000
fixed-context-parameters = {chirp_mass_proxy: 1.19786}
# Alternatively, determine the chirp mass from the data:
# chirp-mass-scan = true

importance-sample = true
importance-sampling-settings = {synthetic_phase: {co_rotate_spins: true, n_grid: 5001, uniform_weight: 0.01}}

################################################################################
## Data generation arguments
################################################################################

trigger-time = 1187008882.4
label = GW170817
outdir = outdir_GW170817
channel-dict = {H1:GWOSC, L1:GWOSC, V1:GWOSC}
psd-length = 128
```

Importance sampling follows the standard [workflow](result.md). For a multibanded
model the likelihood is evaluated on the undecimated base domain by default
(`use_base_domain`, set automatically and adjustable in
`importance-sampling-settings`). Phase-marginalized networks reconstruct the phase
synthetically before reweighting; for BNS models the recommended setting is
`co_rotate_spins: true`, an exact phase draw at one waveform evaluation per sample.
See [synthetic phase](result.md#synthetic-phase) for the alternatives.

As an indication of expected performance, analyses of GW170817 on public data with a
development network reach sample efficiencies of roughly 10% and a log Bayes factor
relative to noise of about +500. A scan run recovers the trigger chirp mass and
matches the evidence of the pinned run within Monte Carlo uncertainty, as expected
from the prior-conditioning construction.

```{eval-rst}
.. footbibliography::
```
