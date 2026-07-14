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
This page describes inference with a trained chirp-mass-conditioned network.
Training such a network is not yet supported; the training configuration will be
documented when it is, and no pre-trained BNS network is distributed yet. Also not
yet available: the accelerated heterodyned and decimated likelihood of
{footcite:p}`Dax:2024mcn` (importance sampling uses the exact likelihood), the
synthetic-phase `compute_likelihood` fast path, and per-event time scans for
pre-merger networks.
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
[sampler context](sampling_chains.md#the-sampler-context) reads this to prepare data
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

The pinned values have a single owner, the chain root, and are recorded with the
samples. The heterodyne receives the proxy through the row-aligned `prepared_data`
contract of the [sampler context](sampling_chains.md#the-sampler-context). Since the
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
the kernel width, draws a few posterior samples at each grid point in a single
batched network pass, evaluates a phase-marginalized likelihood for every draw within
the prior, and takes the chirp mass of the maximum-likelihood draw as the trigger
value.

The sweep runs on the ordinary chain machinery: a fixed table with one row per grid
point roots the chain, the data preparation heterodynes each row at its own proxy
value, and the network draws per row. The scan result (trigger value, signal-to-noise
ratio, maximum log likelihood, and the scan settings) is recorded in the sampler
provenance. For a GW170817-like event the scan costs about a minute of CPU time.

## Running through dingo_pipe

Two [dingo_pipe](dingo_pipe.md) sampler options control BNS inference:

fixed-context-parameters
: Dictionary pinning the model's context parameters, e.g.
  `{chirp_mass_proxy: 1.19786, ra: 3.44616, dec: -0.408084}`. A single-network model
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
fixed-context-parameters = {chirp_mass_proxy: 1.19786, ra: 3.44616, dec: -0.408084}
# Alternatively, determine the chirp mass from the data:
# chirp-mass-scan = true
# fixed-context-parameters = {ra: 3.44616, dec: -0.408084}

importance-sample = true
importance-sampling-settings = {synthetic_phase: {approximation_22_mode: true, n_grid: 5001, uniform_weight: 0.01}}

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
synthetically before reweighting; setting `approximation_22_mode: true` treats the
signal as dominated by the $(2, 2)$ mode, which is appropriate for BNS and
substantially faster than the mode-summed default.

As an indication of expected performance, analyses of GW170817 on public data with a
development network reach sample efficiencies of roughly 10% and a log Bayes factor
relative to noise of about +500. A scan run recovers the trigger chirp mass and
matches the evidence of the pinned run within Monte Carlo uncertainty, as expected
from the prior-conditioning construction.

```{eval-rst}
.. footbibliography::
```
