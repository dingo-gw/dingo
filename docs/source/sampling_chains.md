# Sampling chains

## Introduction

In practice, obtaining posterior samples is more complicated than just sampling from a flow. It is often also necessary, for instance, to add fixed parameters, apply coordinate transformations (sky rotations), iterate with GNPE, or reconstruct the synthetic phase (which requires access to the likelihood). The additional steps necessary for sampling in these cases are determined by model and event metadata. However, without a systematic organizing principle, the code to implement these steps can become unwieldy.

Dingo's approach is the **factorized sampler**, which organizes sampling into an explicit chain of steps. The idea is to express the posterior as an ordered product of conditionals,

$$
q(\theta_1, \ldots, \theta_n | d) = \prod_i q_i(\theta_i | \theta_{<i}, d).
$$

Each conditional $q_i$ in this product is one step of the chain: a normalizing flow, for instance, a point mass that pins parameters to fixed values, or the phase posterior conditioned on the remaining parameters. A step may also be a deterministic change of variables, such as the sky rotation between reference frames.

The chain acts on a table of named parameters, together with the log probability. Each step adds columns to the table, or replaces existing columns with transformed ones, and contributes a term to the log probability of the samples,

$$
\log q(\theta | d) = \sum_i \Delta_i,
$$

where $\Delta_i$ is the log density of a stochastic step (identically zero for a point mass) or the Jacobian term $-\log\lvert\det J_i\rvert$ of a change of variables. The [step types](#steps) below make this precise. (The log probability becomes the proposal density when importance sampling.) Note that the Gibbs sampling of GNPE breaks access to the density; see below.

Two example chains:
* **Plain NPE** (`FlowFactor → RAToEventFrame`): the flow network, followed by a rotation
  of the right ascension from the training reference frame to the event frame.
* **[DINGO-BNS](bns.md) prior conditioning** (`DeltaFactor →
  FlowFactor → ProxyOffsetReparam → RAToEventFrame`): the chirp mass proxy value is
  pinned, which conditions the network, and the network's offset output is
  then combined with the proxy to reconstruct the physical chirp mass.

The `ChainComposer` class holds the chain of steps and carries out sampling. When a chain is constructed, the composer checks it for consistency: every conditioning column must be produced by an earlier step. When sampling, the composer runs the steps in order, building up the table and the log-density sum:

```{mermaid}
:caption: Sampling from a chain of steps. Each step contributes its term $\Delta_i$ to the running log probability.

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

In the figure, every step is written as a conditional $q_i$, point masses included; a change of variables contributes its Jacobian term in place of a log density.

The generic machinery is defined in `dingo.core.inference`: `steps` (the step
types), `composer` (the composer, the Gibbs block, and the runner), and `context` (the
context protocol). The gravitational-wave steps, the per-event context, and the
chain builders are defined in `dingo.gw.inference`. The builders described in
[](inference.md) assemble the standard chains from model metadata. A chain is
ordinary Python, however, and can just as well be assembled by hand (see
[](#building-and-running-a-chain)).

## Steps

Each entry in a chain is a step. A step is an object with `parameters` (the
columns it emits), `conditioning` (the earlier columns it reads), `produces` (every
column it adds, `parameters` plus any side channels), `draws` (whether it draws
samples), and a `sample_and_log_prob` method; together these form the `Step`
protocol. Steps never
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

A `Reparametrization` is a deterministic bijection. Its `forward` method maps its
`inputs` (and any read-only `conditioning`) to new columns, which replace the
inputs in the chain. Its `inverse` method rebuilds the inputs. The proposal density gains a term
$-\log\lvert\det J\rvert$. A reparametrization is 1:1, with one output row per input row, so it
carries no sample multiplicity.

* `RAToEventFrame` rotates the right ascension from the network's training
  reference frame (`ra@t_ref`) to the event frame (`ra`). The rotation angle is the
  sidereal-time difference between the event time and the training reference time,
  which is exactly zero when the two times are equal. `RAToTrainingFrame` applies
  the same rotation in the opposite direction, which is useful when pinning the sky position for a sky-conditional network.
* `ProxyOffsetReparam` reconstructs a physical parameter from a network's offset
  output and its proxy, $X = \delta_X + X_\mathrm{proxy}$. The offset is its input,
  replaced by $X$; the proxy is read-only conditioning and stays in the chain. This is used when prior-conditioning BNS inference on the chirp mass.
* `SpinConventionReparam` relabels the precessing-spin angles between Dingo's
  internal spin-phase convention and that used by Bilby.

### Target corrections

A `TargetCorrection` emits an annotation column, `delta_log_prob_target`. During
importance sampling, this column is added to the *target* log density. The step
contributes nothing to the proposal. Target corrections cover cases where the
target is not simply $\pi(\theta)\,\mathcal{L}(\theta)$. The emitted column is an
annotation rather than a parameter block, so the step adds no conditional to the
product, and its proposal term is $\Delta_i = 0$. The one instance is
`GNPEKernelCorrection`. In single-step GNPE, the proposal is the joint
$q(\hat\theta)\,q(\theta | d, \hat\theta)$ over parameters and proxies, and the
matching target then includes the kernel term $p(\hat\theta | \theta)$. This term
is evaluated at the detector times recomputed from $\theta$, and the result is
recorded with the samples. The recomputed detector times are a *side channel* of the
main network: a column a step emits beyond its `parameters`, which later steps may
read but which is not part of the chain's output. A target correction has no
inverse, so `ChainComposer.log_prob` skips it.

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
  and cached. When called with conditioning, the result has one data row per conditioning row. As an example, for DINGO-BNS, the `chirp_mass_proxy` parameter should be provided as conditioning, and `prepared_data()` will use this for heterodyning. For a tokenized (transformer) network the representation is the list `[waveform, position, token_mask]`, and a frequency-range update or PSD notch masks tokens rather than bins.

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
* A *duration or frequency-range update for importance sampling* regenerates the
event data on the requested grid, wider than the network's band if asked, which
the pipe records with them; the likelihood works on that grid. For a multibanded
model, whether the likelihood uses the base domain or the decimated bands is the
`use_base_domain` argument of `likelihood()`; data off the network's grid (a wider
range or another duration) require the base domain.
* A *prior update* is applied at the
importance-sampling stage and never modifies the context.

A context is built with `GWSamplerContext.from_model(model, event_data,
event_metadata)`. It can also be built from a metadata dictionary alone, with
`from_model_metadata`. A saved `Result` uses this route to reconstruct the prior,
domain, and likelihood from its stored settings. The chain's torch device is
`context.device`. Steps that create fresh tensors, such as the pins of a
`DeltaFactor`, create them on this device, so that their outputs can join a chain
running on a GPU.

A context is **immutable**: it is built once from an event dataset and the model metadata. To analyze the same event with different data (for importance sampling), build a new context from the new event dataset. The parameters keep their meaning, so samples drawn under one context can be importance sampled under another.

```{note}
The representation vocabulary in this section (frequency domains, multibanded
decimation, base-domain likelihoods) is specific to this domain family. To support
a new domain family, write a new context class implementing the same interface
(the `dingo.core.inference.context.SamplerContext` protocol: `prepared_data` and `likelihood`, the `event_metadata` and `device` attributes, plus the `prior` used by importance sampling) rather than extending this one.
```

## Sampling mechanics

### Sample counts and multiplicity

The `ChainComposer` orchestrates passes through the chain to obtain samples and/or log probabilities. This includes managing batching, nontrivial sampling multiplicity, and ensuring consistency of the DAG. It holds the steps as an ordered list, validated at construction to ensure it satisfies the topological order of the conditioning DAG: every conditioning column must be produced by an earlier step, and no factor may overwrite an existing column. A reparametrization, however, may replace its own inputs.

Sampling runs the steps in order over a growing table of rows. Only some steps *draw*: a `FlowFactor` or a `GibbsBlock` produces new random values, whereas a `DeltaFactor`, a `SampleTableFactor`, a reparametrization, or a target correction emits exactly one row for each row it receives (the `draws` attribute of a step records which). Each drawing step is given a *count*, the number of samples it draws for each row of the table so far, and the rows already in the table are repeated to match, so that every row stays complete. The total number of samples produced by the chain is therefore

$$
\text{(total samples)} = \text{(root rows)} \times \prod_{\text{drawing steps } i} n_i .
$$

Here, the *root rows* are the rows the table starts with: one, unless the chain is rooted in a `SampleTableFactor`, in which case it starts with the rows of that table. (A chain with no drawing step at all, such as a stored table run through a reparametrization, has an empty product and simply emits its root rows once.) The counts $n_i$ are the `num_samples` argument of `ChainComposer.sample_and_log_prob(num_samples, context, batch_size)`. In the usual case this is an int: the count for the first drawing step (typically a `FlowFactor`), after which every later drawing step draws one sample per row. A sequence of ints instead gives one count per drawing step, in chain order, so that a later step may draw several samples for each row it receives. This would allow, for example, several extrinsic-parameter draws for each intrinsic sample.

The reason to draw only at the point of sampling, rather than repeating a pinned root `num_samples` times up front, is to avoid redundant calculations. For instance, in the DINGO-BNS chirp-mass scan, the `SampleTableFactor` emits a column vector of `chirp_mass_proxy` values, along a grid spanning the prior. For each of these, we prepare one set of heterodyned data. Then the flow draws `num_samples` samples (typically 10) for each grid point. By having the flow perform the expansion (rather than doing it earlier) we avoid redundant data preprocessing and embedding network passes.

`batch_size` splits the first count into chunks, which caps the peak memory at one chunk. For a chain rooted in a `SampleTableFactor`, each chunk still runs over the whole table, so a caller with a large table, such as the chirp-mass scan, splits the table into blocks itself.

### Provenance

A `Result` exported from a composed sampler records how its samples were made,
under `settings["sampler"]`. The record lists the executed chain in order, with one
descriptor per step, and the `num_samples` requested, plus any entries added by
the caller. For example, `dingo_pipe` adds the model checkpoint paths (`models`),
the seed it used (`sampling_seed`), the density-recovery recipe, and the
chirp-mass-scan record.

```python
{"chain": [
     {"step": "DeltaFactor",
      "parameters": ["chirp_mass_proxy", "ra", "dec"], "conditioning": [],
      "values": {"chirp_mass_proxy": 1.1976, "ra": 3.446, "dec": -0.408}},
     {"step": "RAToTrainingFrame", ...},
     {"step": "FlowFactor", ...},
 ],
 "num_samples": 50000,
 "models": {"model": "model.pt"},
 "sampling_seed": 12345}
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
  `DeltaFactor` of fixed detector-time proxies, or an unconditional NDE for
  [density recovery](result.md#density-recovery)). The main network and the kernel
  correction follow it.

A chain is ordinary Python, and the builders use the same pieces that are available
to you:

```python
from dingo.core.inference.composer import ChainComposer
from dingo.core.inference.steps import FlowFactor
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import RAToEventFrame

context = GWSamplerContext.from_model(model, event_data, event_metadata)
flow = FlowFactor(model, aliases={"ra": "ra@t_ref"})
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
   `conditioning` (the columns read). A reparametrization also sets `inputs`, the
   columns it transforms, which its outputs replace; its `conditioning` is then
   the read-only remainder. A factor that emits columns beyond `parameters`
   declares them in `produces`; such side channels are intermediates for later
   steps and are dropped from the output. A target correction samples nothing:
   its `parameters` are empty, and it declares its annotation column in `produces`.
   A factor that does not draw new samples (a point mass, a fixed table) sets
   `draws = False`, so that the composer runs it once rather than asking it for
   `num_samples`.
3. **Implement the contract.**
   * A factor implements `sample_and_log_prob` and `log_prob`, both in physical
     parameter space. `sample_and_log_prob` returns `num_samples` draws per
     conditioning row, with the draws for a given row adjacent.
   * A reparametrization implements `forward` and `inverse` (and `log_det` when
     the map is not measure-preserving). The inverse must rebuild exactly the
     `inputs`, since `ChainComposer.log_prob` relies on it to restore them.
   * A target correction implements `correction`.
4. **Read data only through the context.** This keeps the step valid under
   another context for the same event.
5. **Override `describe()`** if the step has configuration worth recording, and
   keep the descriptor literal-only.

## API

The classes on this page are documented in the API reference:

* {py:class}`dingo.core.inference.steps.Factor`, with {py:class}`~dingo.core.inference.steps.FlowFactor`, {py:class}`~dingo.core.inference.steps.DeltaFactor`, and {py:class}`~dingo.core.inference.steps.SampleTableFactor`
* {py:class}`dingo.core.inference.steps.Reparametrization`
* {py:class}`dingo.core.inference.steps.TargetCorrection`
* {py:class}`dingo.core.inference.composer.GibbsBlock`
* {py:class}`dingo.core.inference.composer.ChainComposer`
* {py:class}`dingo.gw.inference.context.GWSamplerContext`, and the gravitational-wave steps in {py:mod}`dingo.gw.inference.steps`
