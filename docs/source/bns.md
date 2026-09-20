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
is in `examples/binary_neutron_stars`, see the [example](example_bns.md)), the
accelerated heterodyned and decimated likelihood of {footcite:p}`Dax:2024mcn`
(importance sampling uses the exact likelihood), the synthetic-phase
`compute_likelihood` fast path, and per-event time scans for pre-merger networks.
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

A network may condition on further context parameters, such as the sky position,
and any of them can be pinned in the same way; the frame handling of a pinned right
ascension is described under [sampling chains](sampling_chains.md#steps).

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

## Multibanding heterodyned data

The bands of the multibanded frequency domain are chosen by the
[band tool](waveform_dataset.ipynb#generating-a-multibanded-domain), which decimates
test waveforms until their mismatch with the uniform-domain waveforms meets a target. Heterodyning changes what the tool has to look at. Decimation does not
commute with heterodyning, so the bands must be chosen on heterodyned waveforms, as the
network sees them; when the dataset settings contain `phase_heterodyning`, the tool
heterodynes the waveforms first. Moreover, the network only ever sees data heterodyned
at the proxy, which can sit anywhere in the kernel around the true chirp mass, and the
oscillation left after heterodyning depends on which side it sits: the offset term of
the phase flips sign with the offset and adds to or cancels against the
post-Newtonian remainder, so the two sides of the kernel leave different residuals. The
tool therefore heterodynes alternate waveforms at the two edges of the kernel, the
chirp mass plus and minus `--chirp_mass_proxy_offset`, and the mismatch target holds on
the worse side. The offset is set to the half-width of the training kernel; the
[example](example_bns.md) shows the command.

## Tidal approximants

Tidal approximants such as `IMRPhenomXP_NRTidalv3` run through the standard LAL
`WaveformGenerator` in both the uniform and the multibanded frequency domain: the
tidal deformabilities `lambda_1` and `lambda_2` are inserted into the LAL parameter
dictionary whenever present, and they are ordinary inference parameters, listed with
the others (see the [example](example_bns.md)). The network is phase marginalized, so
the phase is reconstructed synthetically before importance sampling; for these models
the synthetic phase should use `co_rotate_spins: true`, and the
[synthetic phase](result.md#synthetic-phase) section explains why.

```{eval-rst}
.. footbibliography::
```
