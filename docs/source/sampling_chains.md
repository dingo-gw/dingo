# Sampling chains

## Introduction

In practice, obtaining posterior samples is more complicated than just sampling from a flow. It is often also necessary, for instance, to add fixed parameters, apply coordinate transformations (sky rotations), iterate with GNPE, or reconstruct the synthetic phase (which requires access to the likelihood). The additional steps necessary for sampling in these cases are determined by model and event metadata. However, without a systematic organizing principle, the code to implement these steps can become unwieldy.

Dingo's approach is the **factorized sampler**, which organizes sampling into an explicit chain of steps. The idea is to express the posterior as an ordered product of conditionals,

$$
q(\theta_1, \ldots, \theta_n | d) = \prod_i q_i(\theta_i | \theta_{<i}, d).
$$

A **factor** in this product represents a step, which could include, e.g., a normalizing flow, a Dirac delta function, or the phase posterior conditioned on the remaining parameters. In general a factor may be stochastic, such as a network, or it may be a point mass that pins parameters to fixed values. In addition to factors, there can also be **reparametrization** steps (e.g., the sky rotation).

The chain of steps acts on a table of named parameters along with the log probability, and the table is modified as it is operated on by each step. Factors add additional columns (parameters), whereas reparametrization steps replace columns with transformed ones. Both contribute to the log probability of the samples,

$$
\log q(\theta | d) = \sum_i \Delta_i,
$$

where $\Delta_i = \log q_i$ for a factor (identically zero for a point mass) and
$-\log\lvert\det J_i\rvert$ for a reparametrization. (The log probability becomes the proposal density when importance sampling.) Note that the Gibbs sampling of GNPE breaks access to the density; see below.

Two example chains:
* **Plain NPE** (`FlowFactor → RAToEventFrame`): the flow network, followed by a rotation
  of the right ascension from the training reference frame to the event frame.
* **[DINGO-BNS](bns.md) prior conditioning** (`DeltaFactor →
  FlowFactor → ProxyOffsetReparam → RAToEventFrame`): the chirp mass proxy value is
  pinned, which conditions the network, and the network's offset output is
  then combined with the proxy to reconstruct the physical chirp mass.

The `ChainComposer` class holds the chain of steps and carries out sampling. When a chain is constructed, the composer checks it for consistency: every conditioning column must be produced by an earlier step. When sampling, the composer runs the steps in order, building up the table and the log-density sum:

```{mermaid}
:caption: Sampling from a chain of factors. This figure omits reparametrization steps, which replace existing columns and contribute a Jacobian term to the log density.

flowchart TB
    subgraph comp ["ChainComposer"]
        direction TB
        s1["step 1<br/>q#8321;(#952;#8321; | d)"]
        s2["step 2<br/>q#8322;(#952;#8322; | #952;#8321;, d)"]
        dots["#8230;"]
        sn["step n<br/>q#8345;(#952;#8345; | #952;#8321;, #8230;, #952;#8345;#8331;#8321;, d)"]
        s1 -- "#916;#8321;" --> s2
        s2 -- "#916;#8321; + #916;#8322;" --> dots
        dots --> sn
    end
    out(["samples + log_prob"])

    sn -- "#916;#8321; + #8230; + #916;#8345; = log q" --> out

    classDef step fill:#dbe9f6,stroke:#2980b9,color:#1a1a1a
    classDef ghost fill:none,stroke:none
    class s1,s2,sn step
    class dots ghost
```

In the figure, every step is written as a conditional $q_i$, which covers all factors, point masses included.

The generic machinery (steps, stages, the composer, and the runner) is defined in
`dingo.core.factors`. The gravitational-wave steps, the per-event context, and the
chain builders are defined in `dingo.gw.inference`. The builders described in
[](inference.md) assemble the standard chains from model metadata. A chain is
ordinary Python, however, and can just as well be assembled by hand (see
[](#building-and-running-a-chain)).

## Steps

Each entry in a chain is a step. A step is an object with `parameters` (the
columns it emits), `conditioning` (the earlier columns it reads), and a
`sample_and_log_prob` method; together these form the `Step` protocol. Steps never
receive event data directly. Instead, the data enters through the shared
[sampler context](#sampler-context). There are three types of step, plus one
density-free sampling block:

| Step type           | Emits                                | Log-prob term $\Delta_i$                 | Examples                                                       |
|---------------------|--------------------------------------|------------------------------------------|----------------------------------------------------------------|
| `Factor`            | a sampled parameter block            | its conditional log density               | `FlowFactor`, `DeltaFactor`, `SampleTableFactor`               |
| `Reparametrization` | a deterministic transform of columns | $-\log \lvert \det J \rvert$ (usually 0)  | `RAToEventFrame`, `ProxyOffsetReparam`, `SpinConventionReparam` |
| `TargetCorrection`  | a target-side annotation column      | 0                                         | `GNPEKernelCorrection`                                         |
| `GibbsBlock`        | the blocks of an internal Gibbs loop | none (the chain becomes density-free)     | multi-iteration [GNPE](gnpe.md)                                |

### Factors

A `Factor` is a conditional distribution $q_i(\theta_i | \theta_{<i}, d)$ over one
parameter block. It draws `num_samples` samples *per conditioning row* and returns
its own log density. Note that network standardization is applied internally, so the `Factor` interface is in physical parameter space.

`FlowFactor`
: Wraps a posterior model (NPE flow, FMPE, ...). Three types of model are supported: (1) an *unconditional* model (for example a density-recovery NDE) takes no
  input at all; (2) a *data-conditional* model draws from shared `prepared_data`; and (3) a model with `context_parameters` (GNPE proxies, prior-conditioning pins)
  additionally conditions on chain columns. For models with nontrivial `context_parameters`, the data may be transformed according to their values. A factor may also expose a trained
  parameter name under an alias (`ra → ra@t_ref`), so that a downstream step can
  convert reference frames by name.

`DeltaFactor`
: A point mass $\delta(\theta_i - c)$ that pins parameters to fixed values. A `DeltaFactor` is used in two ways. As the chain root for prior
  conditioning, it supplies the pins that later factors condition on. As a filler,
  it supplies delta-prior parameters that the network does not infer. The `DeltaFactor`
  contributes zero log probability, as we take the total log probability to include only the parameters sampled over; this doesn't affect importance sampling as the same factor appears in numerator and denominator.

`SampleTableFactor`
: A chain root that emits a fixed table of existing samples, together with their
  stored log probability. Use this factor to continue a chain from samples drawn
  earlier (e.g., a chain adding the synthetic phase to previously-drawn samples, or for a BNS chirp-mass scan).

`SyntheticPhaseFactor`, `GNPEKernelFactor`, `GNPEFlowFactor`
: The gravitational-wave factors, defined in `dingo.gw.inference.steps`.
  `SyntheticPhaseFactor` reconstructs the phase for a phase-marginalized network
  from the likelihood (see [synthetic phase](result.md#synthetic-phase)).
  `GNPEKernelFactor` is the GNPE blur kernel $p(\hat\theta | \theta)$.
  `GNPEFlowFactor` is the GNPE main network, which is conditioned on the proxies.

### Reparametrizations

A `Reparametrization` is a deterministic bijection. Its `forward` method maps the
conditioning columns to new columns, replacing the inputs it `consumes`. Its
`inverse` method rebuilds those inputs. The proposal density gains a term
$-\log\lvert\det J\rvert$. A reparametrization is 1:1, with one output row per input row, so it
carries no sample multiplicity.

* `RAToEventFrame` rotates the right ascension from the network's training
  reference frame (`ra@t_ref`) to the event frame (`ra`). The rotation angle is the
  sidereal-time difference between the event time and the training reference time,
  which is exactly zero when the two times are equal. `RAToTrainingFrame` applies
  the same rotation in the opposite direction, which is useful when pinning the sky position for a sky-conditional network.
* `ProxyOffsetReparam` reconstructs a physical parameter from a network's offset
  output and its proxy, $X = \delta_X + X_\mathrm{proxy}$. It consumes the offset
  column and keeps the proxy in the chain. This is used when prior-conditioning BNS inference on the chirp mass.
* `SpinConventionReparam` relabels the precessing-spin angles between Dingo's
  internal spin-phase convention and that used by Bilby.

### Target corrections

A `TargetCorrection` emits a side-channel column, `delta_log_prob_target`. During
importance sampling, this column is added to the *target* log density. The step
contributes nothing to the proposal. Target corrections cover cases where the
target is not simply $\pi(\theta)\,\mathcal{L}(\theta)$. The emitted column is an
annotation rather than a parameter block, so the step adds no conditional to the
product, and its proposal term is $\Delta_i = 0$. The one instance is
`GNPEKernelCorrection`. In single-step GNPE, the proposal is the joint
$q(\hat\theta)\,q(\theta | d, \hat\theta)$ over parameters and proxies, and the
matching target then includes the kernel term $p(\hat\theta | \theta)$. This term
is evaluated at the detector times recomputed from $\theta$, and the result is
recorded with the samples. A target correction has no inverse. It may therefore
consume only side-channel intermediates, never sampled parameters. (A
reparametrization may consume sampled parameters because its `inverse` can rebuild
them.)

### Density-free blocks

A `GibbsBlock` runs blocked Gibbs sampling internally. It seeds the loop from an
init factor, then iterates through its factor list `num_iterations` times. It yields
no log probability, since the cyclic dependency has no tractable marginal. A chain
that contains a `GibbsBlock` is therefore density-free (no log probability). For importance sampling, the density must be [recovered](result.md#density-recovery)
after Gibbs sampling. Dingo uses this step only for multi-iteration [GNPE](gnpe.md).

## Sampler context

The `GWSamplerContext` holds event data and metadata, along with model metadata, and generates derived objects such as preprocessed data for networks and the GW likelihood. The context is shared across steps for consistency.

### Outputs

`prepared_data(conditioning=None)`
: The network-input representation of the event. This takes the raw data, and applies transformations based on the model metadata, e.g., heterodyning, decimating, whitening, frequency-masking, and repackaging. Optional `conditioning` allows for the result to depend on the conditioning parameters. When called without conditioning, the representation is computed once
  and cached. When called with conditioning, the result has one data row per conditioning row. As an example, for DINGO-BNS, the `chirp_mass_proxy` parameter should be provided as conditioning, and `prepared_data()` will use this for heterodyning.

`prior`
: The prior used for training the network. Importance-sampling prior updates
  and any time/phase split-off for marginalized networks are applied downstream,
  not in the context.

`likelihood(...)`
: The GW likelihood for the event. It is used by likelihood-based
  factors (synthetic phase) and by importance sampling. The likelihood builds its
  own data representation rather than reusing the network-input view. Its
  reference time is the event time when the event metadata provides one, and the
  training reference time otherwise. Marginalization settings (time, phase,
  calibration) are passed with each request.

The figure below shows the data flow:

```{mermaid}
:caption: Data flow through the sampler context. This figure omits likelihood-based factors such as the synthetic phase.

flowchart TB
    d[("event data d")]
    em["event metadata"]
    md["model metadata"]
    ctx["<b>GWSamplerContext</b><br/><i>prepared_data&nbsp;·&nbsp;prior&nbsp;·&nbsp;likelihood</i>"]
    comp["ChainComposer"]
    out(["samples + log_prob"])
    isamp["importance sampling"]

    d --> ctx
    em --> ctx
    md --> ctx
    ctx -- "prepared data,<br/>metadata" --> comp
    comp -. "conditioning" .-> ctx
    comp --> out
    out --> isamp
    ctx -. "likelihood, prior" .-> isamp

    classDef ctxstyle fill:#f4f4f4,stroke:#8c8c8c,color:#1a1a1a
    classDef ctxnode fill:#fff8de,stroke:#b5a642,stroke-width:1.5px,color:#1a1a1a
    classDef step fill:#dbe9f6,stroke:#2980b9,color:#1a1a1a
    class d,em,md ctxstyle
    class ctx ctxnode
    class comp step
```

Event metadata carried by the context includes the event time (used by `RAToEventFrame` and the likelihood) together with any per-event
analysis settings. This allows for some settings to be changed at inference time, e.g.,

* A *frequency-range update* allows for per-detector minimum or maximum frequencies. The update is validated against the training-time random
strain cropping, which must cover the requested range. The likelihood applies the
same range independently, through ASD masking.
* A *representation update* (an
updated duration or choice of domain for importance sampling) produces a
derived context, described below.
* A *prior update* is applied at the
importance-sampling stage and never modifies the context.

A context is built with `GWSamplerContext.from_model(model, event_data,
event_metadata)`. It can also be built from a metadata dictionary alone, with
`from_model_metadata`. A saved `Result` uses this route to reconstruct the prior,
domain, and likelihood from its stored settings. The chain's torch device is
`context.device`. Steps that create fresh tensors, such as the pins of a
`DeltaFactor`, create them on this device, so that their outputs can join a chain
running on a GPU.

A context is treated as **immutable**. To change the representation (e.g., to change the duration), use the `derive()` method to generate a new context. The derived context shares the event data and metadata
with the original, but with updated `domain`, `use_base_domain`, or `wfg_delta_f` values. Samples drawn under the original context can therefore be importance sampled under the derived one.

```{note}
The representation vocabulary in this section (frequency domains, multibanded
decimation, base-domain likelihoods) is specific to this domain family. To support
a new domain family, write a new context class implementing the same interface
(the `dingo.core.factors.SamplerContext` protocol: `prepared_data` and `likelihood`, plus the `prior` and `derive` methods used by importance sampling) rather than extending this one.
```

## Sampling mechanics

### Stages, fan-out, and multiplicity

The `ChainComposer` orchestrates passes through the chain to obtain samples and/or log probabilities. This includes managing batching, nontrivial sampling multiplicity, and ensuring consistency of the DAG. It represents the chain as an ordered list of `Stage(step, fan_out)` entries, where `fan_out` allows for multiple output samples per input sample at a given `step` (see below). Bare steps are accepted as well, and are wrapped as stages with `fan_out=1`. The stage list is validated at construction to ensure it satisfies the topological order of the conditioning DAG: every conditioning column must be produced by an earlier
step, and no factor may overwrite an existing column. A reparametrization, however, may replace its own inputs.

When running `ChainComposer.sample_and_log_prob(num_samples, context, batch_size)`, the total number of samples produced by the chain is

$$
\text{(total samples)} = \text{(root rows)} \times \texttt{num_samples} \times \prod_{\text{stages } i} \texttt{fan_out}_i .
$$

Here, the *root rows* are the rows the table starts with: one, unless the chain is rooted in a `SampleTableFactor`, in which case it starts with the rows of that table. The argument `num_samples` is used exactly once, by the first step that actually samples (typically a `FlowFactor`). It then produces `num_samples` samples for each row of the table it receives. A stage with `fan_out=k` draws $k$ further samples for each row it receives, multiplying the table by $k$. This is useful, for example, to draw several extrinsic-parameter samples for each intrinsic sample.

The reason to consume `num_samples` only at the point of sampling is to avoid redundant calculations. For instance, in the DINGO-BNS chirp-mass scan, the `SampleTableFactor` emits a column vector of `chirp_mass_proxy` values, along a grid spanning the prior. For each of these, we prepare one set of heterodyned data. Then the flow draws `num_samples` samples (typically 10) for each grid point. By having the flow perform the expansion (rather than doing it earlier) we avoid redundant data preprocessing and embedding network passes.

`batch_size` splits `num_samples` into chunks, which caps the peak memory at one chunk. For a chain rooted in a `SampleTableFactor`, each chunk still runs over the whole table, so a caller with a large table, such as the chirp-mass scan, splits the table into blocks itself.

### Provenance

A `Result` exported from a composed sampler records how its samples were made,
under `settings["sampler"]`. The record lists the executed chain in order, with one
descriptor per step, plus any entries added by the caller. For example,
`dingo_pipe` adds the model checkpoint paths (`models`), the density-recovery
recipe, and the chirp-mass-scan record.

```python
{"chain": [
     {"step": "DeltaFactor",
      "parameters": ["chirp_mass_proxy", "ra", "dec"], "conditioning": [],
      "values": {"chirp_mass_proxy": 1.1976, "ra": 3.446, "dec": -0.408}},
     {"step": "RAToTrainingFrame", ...},
     {"step": "GNPEFlowFactor", ...},
 ],
 "models": {"model": "model.pt"}}
```

This block is a record of what was run. Nothing reads it at load time, and in
particular the chain is not rebuilt from it. The block is also **literal-only**:
every value round-trips through `str`/`ast.literal_eval` in the saved settings.

## Building and running a chain

The standard chains are assembled from model metadata by the `GWComposedSampler`
builders (see [](inference.md) for usage):

`from_model(model, event_data, event_metadata, fixed_context_parameters=None)`
: A single-network chain. This covers plain NPE. It also covers
  [prior conditioning](bns.md) for a model with `context_parameters`, in which case
  the pinned values form the chain root.

`from_gnpe_models(init_model, main_model, event_data, event_metadata, num_iterations=30)`
: Multi-iteration time GNPE. The chain contains a `GibbsBlock`, which cycles the
  kernel and main-network factors. The chain is density-free.

`from_singlestep_gnpe(main_model, proxy_source, event_data, event_metadata)`
: Single-step, density-preserving GNPE. The `proxy_source` supplies the proxies (a
  `DeltaFactor` for prior conditioning, or an unconditional NDE for
  [density recovery](result.md#density-recovery)). The main network and the kernel
  correction follow it.

A chain is ordinary Python, and the builders use the same pieces that are available
to you:

```python
from dingo.core.factors import ChainComposer, FlowFactor
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import RAToEventFrame

context = GWSamplerContext.from_model(model, event_data, event_metadata)
flow = FlowFactor.from_model(model, aliases={"ra": "ra@t_ref"})
composer = ChainComposer([flow, RAToEventFrame()])

samples = composer.sample(10_000, context, batch_size=5_000)
```

This is the chain that `from_model` assembles for a plain NPE model (plus a
`DeltaFactor` filler for any delta-prior parameters). `GWComposedSampler` adds the
DataFrame runner (`run_sampler`) and the `Result` export (`to_result` / `to_hdf5`).

### Writing a new step

1. **Pick the type.** A step that samples new parameters is a `Factor`. A step
   that transforms existing parameters deterministically and invertibly is a
   `Reparametrization`. A step that annotates the importance-sampling target is a
   `TargetCorrection`.
2. **Declare the interface.** Set `parameters` (the columns emitted) and
   `conditioning` (the columns read). Set `consumes` if the step removes columns
   from the chain (a reparametrization's replaced inputs, a correction's
   intermediates), and `produces` if a factor emits columns beyond `parameters`.
   A factor that does not draw new samples (a point mass, a fixed table) sets
   `draws = False`, so that the composer runs it once rather than asking it for
   `num_samples`.
3. **Implement the contract.**
   * A factor implements `sample_and_log_prob` and `log_prob`, both in physical
     parameter space. `sample_and_log_prob` returns `num_samples` draws per
     conditioning row, with the draws for a given row adjacent.
   * A reparametrization implements `forward` and `inverse` (and `log_det` when
     the map is not measure-preserving). The inverse must rebuild exactly the
     consumed columns, since `ChainComposer.log_prob` relies on it to restore
     them.
   * A target correction implements `correction`.
4. **Read data only through the context.** This keeps the step valid under a
   derived context.
5. **Override `describe()`** if the step has configuration worth recording, and
   keep the descriptor literal-only.

## API

The classes on this page are documented in the API reference:

* {py:class}`dingo.core.factors.Factor`, with {py:class}`~dingo.core.factors.FlowFactor`, {py:class}`~dingo.core.factors.DeltaFactor`, and {py:class}`~dingo.core.factors.SampleTableFactor`
* {py:class}`dingo.core.factors.Reparametrization`
* {py:class}`dingo.core.factors.TargetCorrection`
* {py:class}`dingo.core.factors.GibbsBlock`
* {py:class}`dingo.core.factors.Stage` and {py:class}`dingo.core.factors.ChainComposer`
* {py:class}`dingo.gw.inference.context.GWSamplerContext`, and the gravitational-wave steps in {py:mod}`dingo.gw.inference.steps`
