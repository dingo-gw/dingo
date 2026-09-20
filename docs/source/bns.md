# Binary neutron stars

Dingo analyzes binary neutron star (BNS) events with the DINGO-BNS method of
{footcite:p}`Dax:2024mcn`. BNS signals pose two problems for plain NPE. Their long
inspirals require a fine frequency resolution, so the data are far larger than for
binary black holes, and their chirp mass is measured so precisely that a network
covering the full training prior would spend nearly all of its capacity on parameter
values excluded by any individual event. DINGO-BNS addresses both with a single
device: a chirp-mass proxy that simplifies the data (phase heterodyning) and narrows
the effective prior (prior conditioning). The proxy is fixed per event, so sampling is
single-step GNPE: one pass through the network, with the density preserved and
importance sampling available directly.

```{note}
Not yet available: a distributed pre-trained BNS network (the training configuration
is in `examples/binary_neutron_stars`, see [Training](#training)), the accelerated
heterodyned and decimated likelihood of {footcite:p}`Dax:2024mcn` (importance sampling
uses the exact likelihood), the synthetic-phase `compute_likelihood` fast path, and
per-event time scans for pre-merger networks.
```

## Phase heterodyning

At BNS frequency resolutions the strain oscillates rapidly in frequency, which makes
poor input for a network. Multiplying the data by $\exp(i \phi(f; \tilde{\mathcal{M}}))$
with the leading-order chirp phase

$$
\phi(f; \tilde{\mathcal{M}}) = \frac{3}{128}
\left(\frac{\pi G \tilde{\mathcal{M}} f}{c^3}\right)^{-5/3}
$$

removes the dominant phase evolution for a reference chirp mass
$\tilde{\mathcal{M}}$ close to the true value. The residual oscillations are slow,
and the multibanded frequency domain can then decimate the data far more
aggressively.

The reference value is the *chirp-mass proxy*. During training it is drawn by
blurring the true chirp mass with a narrow kernel, and the network conditions on it;
this is GNPE with the chirp mass as the proxy parameter (see [GNPE](gnpe.md)). At
inference the proxy is fixed per event, so a single iteration suffices and the
density is preserved. A chirp-mass-conditioned model records the kernel and the
phase order in its metadata under `gnpe_chirp`, and the
[sampler context](sampling_chains.md#sampler-context) reads this to prepare data
as a function of `chirp_mass_proxy`. Heterodyning is applied to the raw strain before
decimation (the two operations do not commute).

## Prior conditioning

The proxy plays a second role. Because the network conditions on
$\tilde{\mathcal{M}}$, it effectively learns a family of posteriors under narrow
chirp-mass priors centered on the proxy, $q(\theta | d, \tilde{\mathcal{M}})$.
Setting the proxy at inference selects the member of the family appropriate to the
event, so one network amortizes over events while retaining the resolution of an
event-specific narrow prior. The network infers the offset
`delta_chirp_mass` $= \mathcal{M} - \tilde{\mathcal{M}}$ rather than the chirp mass
itself; the chain reconstructs the physical value with a `ProxyOffsetReparam` step.

The chain:

```{mermaid}
flowchart TB
    pins["DeltaFactor<br/>chirp_mass_proxy"]
    flow["FlowFactor<br/>draws delta_chirp_mass, #8230;"]
    off["ProxyOffsetReparam<br/>chirp_mass = delta_chirp_mass + chirp_mass_proxy"]
    out(["samples + log_prob"])
    ctx["GWSamplerContext<br/>heterodyne #8594; decimate #8594; whiten"]

    pins --> flow --> off --> out
    pins -. "chirp_mass_proxy" .-> ctx
    ctx -. "prepared data" .-> flow

    classDef step fill:#dbe9f6,stroke:#2980b9,color:#1a1a1a
    classDef reparam fill:#e2f0e6,stroke:#27ae60,color:#1a1a1a
    classDef ctxstyle fill:#f4f4f4,stroke:#8c8c8c,color:#1a1a1a
    class pins,flow step
    class off reparam
    class ctx ctxstyle
```

The pinned values have a single owner, the chain root, and are recorded with the
samples. The heterodyne receives the proxy through the row-aligned `prepared_data`
contract of the [sampler context](sampling_chains.md#sampler-context). Since the
chain contains no Gibbs block, the samples carry their log probability and
importance sampling proceeds without a density-recovery step.

A network may condition on further context parameters, and any of them can be pinned
in the same way. The reference configuration of {footcite:p}`Dax:2024mcn` also fixes
the sky position: a pinned right ascension is given in the event frame and rotated
into the network's training frame before conditioning, which inserts an
`RAToTrainingFrame` step before the network and a trailing `RAToEventFrame` that
restores the event-frame value in the samples. The rotation is exactly zero when the
network's reference time equals the trigger time.

## The chirp-mass scan

When no external estimate of the chirp mass is available, the trigger value can be
determined from the data (see the Methods of {footcite:p}`Dax:2024mcn`). The scan
sweeps the proxy over the training chirp-mass prior on a grid whose spacing is set by
the kernel width, draws a few posterior samples at each grid point in batched
network passes over blocks of grid points, evaluates a phase-marginalized likelihood for every draw within
the prior (the scan therefore requires a phase-marginalized network), and takes the chirp mass of the maximum-likelihood draw as the trigger
value.

The sweep runs on the ordinary chain machinery: a fixed table with one row per grid
point roots the chain, the data preparation heterodynes each row at its own proxy
value, and the network draws per row. The scan result (trigger value, signal-to-noise
ratio, maximum log likelihood, and the scan settings) is recorded in the sampler
provenance. For a GW170817-like event the scan costs about a minute of CPU time.

## Tidal parameters

Tidal approximants such as `IMRPhenomXP_NRTidalv3` run through the standard LAL
`WaveformGenerator` in both uniform and multibanded frequency domains: `lambda_1`
and `lambda_2` are inserted into the LAL parameter dictionary whenever present.

Beyond a BBH setup, a tidal run needs:

- **Dataset, `waveform_generator`**: the tidal `approximant` and
  `spin_conversion_phase: 0.0` (required for phase-marginalized training).
- **Dataset, `intrinsic_prior`**: BNS-appropriate mass and spin ranges (the
  `default` entries are tuned to BBH), plus `lambda_1: default` /
  `lambda_2: default` (`Uniform(0, 5000)`) or explicit prior strings.
- **Training**: list `inference_parameters` explicitly, including `lambda_1` and
  `lambda_2` (the `default` list contains only the 15 BBH parameters).
- **Importance sampling**: set `co_rotate_spins: true` in the synthetic-phase
  settings (see below).

For the synthetic phase, `co_rotate_spins: true` is recommended: for models with
only a co-precessing $(2, \pm 2)$ pair it is exact at a single waveform
evaluation per sample. It draws the phase in the physical spin convention —
where a phase shift co-rotates the in-plane spins, so the waveform transforms as
a global $e^{2i\phi}$ factor — and rotates `theta_jn` / `phi_jl` accordingly; a
two-waveform probe verifies this property at runtime and falls back to the exact
mode sum otherwise. The plain `approximation_22_mode: true` shortcut has the
same cost but is approximate for precessing signals: precession mixes the
inertial-frame $m$-components, placing a spurious phase peak at $\phi + \pi$. 
The exact mode sum (`approximation_22_mode: false`) costs $2\ell_{\max}+1 = 5$ 
evaluations per sample and, since NRTidal models have no frequency-domain modes in LALSimulation, requires the DFT phase decomposition with an explicit 
`mode_list: [[2, 2], [2, -2]]` in the waveform-generator settings.

## Training

A chirp-mass-conditioned network is trained with the standard tools, with three
additions: the waveform dataset is heterodyned before compression, the multibanded
domain is determined from heterodyned waveforms, and the training transforms apply
chirp-mass GNPE. A complete configuration following the GW170817 network of
{footcite:p}`Dax:2024mcn` is provided in `examples/binary_neutron_stars`: 128 s
segments ($\delta f = 1/128$ Hz), an analysis band from 23 Hz to 1536 Hz, a low-spin
prior ($a \leq 0.05$), 30 million waveforms, and a kernel of half-width
$0.005\,M_\odot$ on the chirp mass.

### Waveform dataset

The dataset settings are first written for the uniform base domain
(`waveform_dataset_settings_ufd.yaml`). Beyond the approximant, priors, and
`spin_conversion_phase: 0.0` of [Tidal parameters](#tidal-parameters), the compression
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

Each waveform is heterodyned at its own chirp mass, after whitening and before the
SVD, so the basis is built from heterodyned waveforms and can compress the long
inspirals. The heterodyne belongs to the internal storage only: the dataset inverts it
on decompression, and the training transforms re-heterodyne at the proxy (below).
`order: 0` removes the leading-order chirp phase of
[Phase heterodyning](#phase-heterodyning), and `chirp_mass` must be an intrinsic
parameter. Since heterodyning must precede decimation, the waveforms have to be
generated directly on the dataset domain: on a multibanded domain this restricts
`phase_heterodyning` to approximants with a frequency-domain implementation, and an
error is raised otherwise.

The multibanded domain is then determined from these settings:

```bash
dingo_generate_multibanded_domain --settings_file waveform_dataset_settings_ufd.yaml \
    --target_median_mismatch 1e-5 --chirp_mass_proxy_offset 0.005 --num_processes N
```

This tunes the decimation to the target median mismatch, using waveforms at the
minimum chirp mass of the prior (the longest signals), and writes
`waveform_dataset_settings_mfd.yaml`: the same settings with a
`MultibandedFrequencyDomain`. When the settings contain `phase_heterodyning`, the bands
are determined from heterodyned waveforms, as the network sees them. The network sees
data heterodyned at the proxy rather than at the true chirp mass, and the residual
oscillation grows with the difference between the two, so `--chirp_mass_proxy_offset`
heterodynes at the chirp mass plus or minus this offset (alternating between the two
sides of the kernel, which decimate differently); set it to the half-width of the
training kernel. `dingo_evaluate_multibanded_domain` accepts the same offset. With the
settings above, the target of $10^{-5}$ reproduces the banding of the network of
{footcite:p}`Dax:2024mcn` (eight bands, about 3700 bins from 20 Hz to 2048 Hz). The
dataset is generated from the multibanded settings file with `dingo_generate_dataset`
as usual.

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

### Noise

The ASD datasets are generated as for any other network (see
[noise dataset](noise_dataset.ipynb)), on the uniform base domain with 128 s segments
(`T: 128.0` in `asd_dataset_settings.yaml`): a fiducial dataset for pre-training and a
full dataset for fine-tuning. They are decimated to the multibanded domain
automatically when training starts.

### Train settings

The `data` section of `train_settings.yaml` configures chirp-mass GNPE and the context
parameters:

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
```

`gnpe_chirp` inserts the chirp-mass GNPE transform after the extrinsic parameters are
sampled. The chirp mass is blurred by the kernel, a Bilby prior on the offset, to give
`chirp_mass_proxy`; the polarizations are heterodyned at the proxy, and the network
conditions on it (`chirp_mass_proxy` is appended to `context_parameters`
automatically). The offset `delta_chirp_mass` is available as an inference parameter
and takes the place of `chirp_mass` in `inference_parameters`; the chain restores
`chirp_mass` at inference (see [Prior conditioning](#prior-conditioning)). The SVD that
seeds the embedding network is built from heterodyned waveforms.

Further context parameters may be listed under `context_parameters` and omitted from
`inference_parameters`; the GW170817 network of {footcite:p}`Dax:2024mcn` conditions
on the sky position in this way, whereas the example infers it. At inference every
context parameter is pinned per event (`fixed-context-parameters` below). The
reference network is phase marginalized: `phase` is not an inference
parameter, and the phase is reconstructed synthetically before importance sampling,
which requires `spin_conversion_phase: 0.0` in the dataset. The remaining settings
(model, training stages, local) follow the standard [training](training.md) layout,
and training runs with `dingo_train` or `dingo_train_condor`.

## Running through dingo_pipe

Two [dingo_pipe](dingo_pipe.md) sampler options control BNS inference:

fixed-context-parameters
: Dictionary pinning the model's context parameters, e.g.
  `{chirp_mass_proxy: 1.19786}`. A single-network model
  with context parameters requires all of them pinned, unless the chirp-mass proxy is
  supplied by the scan. Cannot be combined with `model-init` (iterative GNPE).

chirp-mass-scan
: Set to `true` to determine the trigger chirp mass from the data with defaults
  derived from the model (grid from the training prior and kernel, 10 draws per grid
  point). A dictionary overrides individual settings, e.g.
  `{num_samples: 10, overlap_factor: 2, block_size: 32}`; `num_processes` defaults to
  `request-cpus`. Mutually exclusive with a pinned `chirp_mass_proxy`; the remaining
  context parameters are still supplied via `fixed-context-parameters`.

```{code-block} ini
---
caption: Sampler and data sections of a GW170817 configuration.
---
################################################################################
##  Sampler arguments
################################################################################

model = /path/to/bns_model.pt
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
`co_rotate_spins: true` — an exact phase draw at
one waveform evaluation per sample. See
[Tidal parameters](#tidal-parameters) for the alternatives.

As an indication of expected performance, analyses of GW170817 on public data with a
development network reach sample efficiencies of roughly 10% and a log Bayes factor
relative to noise of about +500. A scan run recovers the trigger chirp mass and
matches the evidence of the pinned run within Monte Carlo uncertainty, as expected
from the prior-conditioning construction.

```{eval-rst}
.. footbibliography::
```
