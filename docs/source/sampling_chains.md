# Sampling chains

Sampling the posterior involves more than a pass through a neural network: fixed
parameters are filled in from the prior, coordinates are rotated between reference
frames, GNPE introduces proxy variables, and a marginalized phase may be reconstructed
from the likelihood. Dingo's *factorized sampler* expresses all of this as an explicit
chain of steps acting on a growing table of named parameter columns. The *factors*
among the steps sample the posterior as an ordered product of conditionals,

$$
q(\theta_1, \ldots, \theta_n | d) = \prod_i q_i(\theta_i | \theta_{<i}, d).
$$

A factor may be stochastic (a network) or a point mass pinning parameters to fixed
values, which are conditioned on rather than integrated over. A *reparametrization*
is a bijective change of variables: it replaces sampled columns with transformed
ones, so its inputs disappear from the table and the density picks up a Jacobian
term rather than a new factor. Sampling records the log density of the final table,
which importance sampling later divides by, and each step contributes one term,

$$
\log q(\theta | d) = \sum_i \Delta_i,
$$

where $\Delta_i = \log q_i$ for a factor (identically zero for a point mass) and
$-\log\lvert\det J_i\rvert$ for a reparametrization. These cases correspond to the
[step types](#steps) below. Two example chains:

* **Plain NPE** (`FlowFactor → RAToEventFrame`): the network, followed by a rotation
  of the right ascension from the training reference frame to the event frame.
* **[DINGO-BNS](bns.md) prior conditioning** (`DeltaFactor → RAToTrainingFrame →
  FlowFactor → ProxyOffsetReparam → RAToEventFrame`): pinned proxy and sky values
  feed a conditioned network, whose offset output is then reconstructed into the
  physical chirp mass.

At construction the composer checks the chain for consistency: every conditioning
column must be produced by an earlier step. It then runs the steps in order,
accumulating the table and the sum:

```{mermaid}
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

In the figure every step is written as a conditional $q_i$, which covers all
factors, point masses included. A reparametrization instead replaces columns in
place, adding its Jacobian term to the sum but no factor to the product (see
[the proposal density](#the-proposal-density-forward-and-reverse)).

The generic machinery lives in `dingo.core.factors` (steps, stages, the composer, the
runner); the gravitational-wave steps, the per-event context, and the chain builders
live in `dingo.gw.inference`. The [](inference.md) builders assemble the standard
chains from model metadata, but a chain is ordinary Python and can equally be
assembled by hand (see [](#building-and-running-a-chain)).

## Steps

A chain entry is a *step*: an object with `parameters` (the columns it emits),
`conditioning` (the earlier columns it reads), and a `sample_and_log_prob` method
(the `Step` protocol). Steps never receive event data directly; the data enters
through the shared [sampler context](#the-sampler-context). There are three types of
step, plus one density-free sampling block:

| Step                 | Emits                                | Log-prob term $\Delta_i$                 | Examples                                                       |
|----------------------|--------------------------------------|------------------------------------------|----------------------------------------------------------------|
| `Factor`             | a sampled parameter block            | its conditional log density               | `FlowFactor`, `DeltaFactor`, `SampleTableFactor`               |
| `Reparametrization`  | a deterministic transform of columns | $-\log \lvert \det J \rvert$ (usually 0)  | `RAToEventFrame`, `ProxyOffsetReparam`, `SpinConventionReparam` |
| `TargetCorrection`   | a target-side annotation column      | 0                                         | `GNPEKernelCorrection`                                         |
| `GibbsBlock`         | the blocks of an internal Gibbs loop | none (the chain becomes density-free)     | multi-iteration [GNPE](gnpe.md)                                |

### Factors

A `Factor` is a conditional distribution $q_i(\theta_i | \theta_{<i}, d)$ over one
parameter block: it draws `num_samples` samples per conditioning row and returns its
own log density. Its interface is in physical parameter space; a network's
standardization is applied internally.

`FlowFactor`
: Wraps a posterior model (NPE flow, FMPE, ...), in three conditioning shapes: an
  *unconditional* model (a density-recovery NDE) takes no input at all; a
  *data-conditional* model draws from the shared `prepared_data()`; and a model with
  `context_parameters` (GNPE proxies, prior-conditioning pins) additionally conditions
  on chain columns. A factor may expose a trained parameter name under an alias
  (`ra → ra@t_ref`), so that a downstream step can convert reference frames by name.

`DeltaFactor`
: A point mass $\delta(\theta_i - c)$ pinning parameters to fixed values, contributing
  zero log probability: the pinned block is conditioned on, not integrated over. Used
  as the chain root for prior conditioning, where later factors condition on the pins,
  and as a filler for delta-prior parameters the network does not infer.

`SampleTableFactor`
: A chain root emitting a fixed table of existing samples, together with their stored
  log probability. This is how a chain continues from earlier samples: synthetic phase
  runs as a chain rooted in the proposal samples, and the chirp-mass scan roots its
  sweep in a table with one row per grid point.

`SyntheticPhaseFactor`, `GNPEKernelFactor`, `GNPEFlowFactor`
: The gravitational-wave factors (`dingo.gw.inference.steps`): the likelihood-based
  phase reconstruction for phase-marginalized networks (see
  [synthetic phase](result.md#synthetic-phase)), the GNPE blur kernel
  $p(\hat\theta | \theta)$, and the proxy-conditioned GNPE main network.

### Reparametrizations

A `Reparametrization` is a deterministic bijection: `forward` maps its conditioning
columns to new ones, replacing the inputs it `consumes`; `inverse` rebuilds them; and
the proposal density picks up $-\log\lvert\det J\rvert$, which is zero for the
measure-preserving maps used here. It is 1:1 (one output row per input row), so it
carries no sample multiplicity.

* `RAToEventFrame` rotates the right ascension from the network's training reference
  frame (`ra@t_ref`) to the event frame (`ra`) by the sidereal-time difference, which
  is exactly zero when the event time equals the training reference time.
  `RAToTrainingFrame` is its input-side mirror: a pinned event-frame sky position is
  presented to the network in the frame it was trained in, and a trailing
  `RAToEventFrame` restores the event-frame value in the samples.
* `ProxyOffsetReparam` reconstructs a physical parameter from a network's offset
  output and its proxy, $X = \delta_X + X_\mathrm{proxy}$, consuming the offset column
  while keeping the proxy in the chain.
* `SpinConventionReparam` relabels the precessing-spin angles between Dingo's internal
  spin-phase convention and the physical (Bilby) one.

### Target corrections

A `TargetCorrection` emits a side-channel column, `delta_log_prob_target`, which
importance sampling adds to the *target* log density; it contributes nothing to the
proposal. This covers cases where the target is not simply
$\pi(\theta)\,\mathcal{L}(\theta)$. Because the emitted column is an annotation
rather than a parameter block, the step stands outside the product of conditionals
and contributes $\Delta_i = 0$. The one instance is `GNPEKernelCorrection`: in
single-step GNPE the proposal is the joint $q(\hat\theta)\,q(\theta | d, \hat\theta)$
over parameters and proxies, so the matching target acquires the kernel term
$p(\hat\theta | \theta)$. The step recomputes the detector times from $\theta$,
evaluates the kernel there, and records the result with the samples. Unlike a
reparametrization it has no inverse, so it may consume only side-channel
intermediates, never sampled parameters.

### Density-free blocks

A `GibbsBlock` runs blocked Gibbs sampling internally, seeding from an init factor and
then sweeping its factor list for `num_iterations` iterations. It yields no log
probability, since the cyclic dependency has no tractable marginal. A chain containing
one is density-free (its samples carry no `log_prob`), and the density must be
[recovered](result.md#density-recovery) afterwards. Dingo uses this only for
multi-iteration [GNPE](gnpe.md).

## The sampler context

Every step reads the event through a shared, per-event `GWSamplerContext`: the data
$d$, the per-event metadata, and everything derived from them. It provides three
views.

`prepared_data(conditioning=None)`
: The network-input representation of the event: heterodyning, decimation, whitening,
  frequency masking, and repackaging, as the model's metadata prescribes. Called
  without conditioning, it is computed once and cached. Called with conditioning (the
  chain columns available to a conditioned factor), it is *row-aligned*, with one data
  row per conditioning row. The context consumes only the columns its preparation is a
  function of (for a heterodyning model, `chirp_mass_proxy`); the remaining columns
  condition the network alone. A constant consumed value (a pinned proxy) is prepared
  once and viewed across the rows; varying values (a sweep) run through the
  batch-native transform chain in one vectorized pass.

`prior`
: The static prior fixed at training time (intrinsic and extrinsic, with Dingo
  defaults), built once from the model metadata. Importance-sampling prior updates and
  the time/phase split-off for marginalized networks are applied downstream, not here.

`likelihood(...)`
: The exact likelihood on the raw event data, used by likelihood-based factors
  (synthetic phase) and importance sampling. It builds its own data representation
  rather than reusing the network-input view, and its reference time is the event
  time. Marginalization settings (time, phase, calibration) are passed per request.

As data flow, including the conditioning columns that a chain feeds back into the
data preparation:

```{mermaid}
flowchart TB
    d[("event data d")]
    em["event metadata"]
    md["model metadata"]
    subgraph ctx ["GWSamplerContext"]
        direction LR
        pd["prepared_data"]
        pr["prior"]
        lk["likelihood"]
    end
    chain["chain steps"]
    isamp["importance sampling"]

    d --> ctx
    em --> ctx
    md --> ctx
    pd -- "data rows (row-aligned)" --> chain
    chain -. "conditioning columns (optional)" .-> pd
    lk -.-> isamp
    pr -.-> isamp

    classDef ctxstyle fill:#f4f4f4,stroke:#8c8c8c,color:#1a1a1a
    class d,em,md,pd,pr,lk ctxstyle
```

Alongside the strain data, the context carries the per-event **event metadata**. This
includes the event time, which sets the likelihood reference time and the sidereal
right-ascension correction (`RAToEventFrame`), together with any per-event analysis
settings. Settings can be changed at inference time in three ways, none of which
touches the trained model. A *frequency-range update* (a per-detector minimum or
maximum frequency in the event metadata) narrows the network-input view; it is
validated against the training-time random strain cropping, which must cover the
request, and the likelihood applies the same range independently through ASD masking.
A *representation update* (an updated duration, or a base-domain likelihood for
importance sampling) produces a derived context, described below. A *prior update* is
applied at the importance-sampling stage and never modifies the context.

A context is treated as **immutable**: the data representation is part of its
identity. Importance-sampling settings updates, such as an updated duration or a
likelihood evaluated on the undecimated base domain, produce a *derived* context via
`derive()`. The derived context shares the event payload and analysis metadata, so
parameter meaning is preserved and importance weights between the two representations
remain well defined.

A context is built with `GWSamplerContext.from_model(model, event_data,
event_metadata)`, or from a metadata dictionary alone with `from_model_metadata`; the
latter is how a saved `Result` reconstructs the prior, domain, and likelihood from its
stored settings. The chain's torch device is `context.device`. Steps that create fresh
tensors, such as the pins of a `DeltaFactor`, create them on this device so that their
outputs can join a chain running on a GPU.

Putting the pieces together:

```{mermaid}
flowchart TB
    d[("event data d")] --> ctx["GWSamplerContext"]
    ctx -- "data views" --> comp["ChainComposer"]
    comp -. "conditioning parameters" .-> ctx
    comp --> out(["samples + log_prob"])
    out --> isamp["importance sampling"]
    ctx -.-> isamp

    classDef ctxstyle fill:#f4f4f4,stroke:#8c8c8c,color:#1a1a1a
    classDef step fill:#dbe9f6,stroke:#2980b9,color:#1a1a1a
    class ctx ctxstyle
    class comp step
```

The chain draws its data views through the context and feeds conditioning parameters
back into the preparation. The likelihood and prior views serve the downstream
importance sampling in the same way, so a derived context re-targets the whole
analysis consistently.

```{note}
The representation vocabulary here (frequency domains, multibanded decimation,
base-domain likelihoods) is specific to this domain family. A new domain family
should be given its own context class implementing the same interface
(`prepared_data` / `prior` / `likelihood` / `derive`, the
`dingo.core.factors.SamplerContext` protocol) rather than extending this one.
```

## Stages, fan-out, and multiplicity

A `ChainComposer` holds an ordered list of `Stage(step, fan_out)` entries (bare steps
are wrapped with `fan_out=1`). The declared order must be a topological order of the
conditioning DAG, which is validated at construction: every conditioning column must
be produced by an earlier step, and no step may overwrite an existing column. The
exception is a reparametrization replacing its own inputs: the map is invertible, so
the overwritten state can be restored when evaluating `log_prob`. A consumed column
leaves the produced set, so a later step may re-emit it (as when `RAToEventFrame`
restores the pinned `ra` that `RAToTrainingFrame` consumed).

Sampling follows three multiplicity rules:

1. **The requested count lands on the first stage that carries multiplicity.** A root
   prefix of point masses (pins) emits a single row each, since $n$ draws from a
   point mass are one atom repeated, and 1:1 steps transform rows in place. The first
   sampling stage then draws the full count in a single conditioned call. In the
   DINGO-BNS chain above, the `DeltaFactor` and the frame rotation produce one row,
   and the flow draws all `num_samples` conditioned on it.
2. **Fan-out multiplies rows.** A non-root stage with `fan_out=k` draws $k$ samples
   per conditioning row, and the carried columns are repeated row-major to stay
   aligned; the chain returns `num_samples × expansion` rows, where the expansion is
   the product of the non-root fan-outs. The
   [chirp-mass scan](bns.md#the-chirp-mass-scan) uses this, with a grid-table root
   followed by `Stage(flow, fan_out=num_samples)` drawing per grid row.
3. **Unconditioned non-root steps fill rather than fan.** A delta filler emits one
   constant per current row.

`batch_size` chunks the requested count, capping peak memory at one chunk. A chain
rooted in a `SampleTableFactor` is fixed-size and runs with `batch_size=None`; a
caller sweeping a table, such as the chirp-mass scan, manages its own blocking of the
grid instead.

## The proposal density: forward and reverse

Sampling folds the chain forward, summing each factor's log density and each
reparametrization's Jacobian term into the proposal log probability, reported in
physical parameter space.

`ChainComposer.log_prob(samples, context)` evaluates the same density at given
samples, as needed to re-evaluate saved samples or to importance sample. The steps are
folded in exact *reverse* chain order, so the columns are restored to the state each
step saw during sampling: a reparametrization rebuilds its consumed inputs via
`inverse` (for example, `ra@t_ref` from the event-frame `ra`), a factor adds its
`log_prob` at the restored conditioning, and a target correction contributes nothing.
The reverse fold is what makes single-step GNPE and prior conditioning
density-preserving end to end.

Two special cases complete the bookkeeping. A point-mass factor contributes zero to
the proposal, and the target conditions on the same pinned value, so the analysis is
conditional on the pins rather than integrating over them. A reparametrization
removes the columns it consumes, which is lawful only for an invertible map: the
change of variables supplies the density on the new columns, and the reverse fold
restores the old ones through `inverse`. This is also why a `TargetCorrection`,
which has no inverse, may consume only side-channel intermediates. A `GibbsBlock`
yields samples whose density is intractable, which is what "density-free" means.

## Provenance

A `Result` exported from a composed sampler records how its samples were made under
`settings["sampler"]`: the executed chain in order, one descriptor per step, plus any
entries added by the caller. `dingo_pipe` records the model checkpoint paths
(`models`), the density-recovery recipe, and the chirp-mass-scan record.

```python
{"version": 1,
 "implementation": "composed",
 "chain": [
     {"step": "DeltaFactor",
      "parameters": ["chirp_mass_proxy", "ra", "dec"], "conditioning": [],
      "values": {"chirp_mass_proxy": 1.1976, "ra": 3.446, "dec": -0.408}},
     {"step": "RAToTrainingFrame", ...},
     {"step": "GNPEFlowFactor", ...},
 ],
 "models": {"model": "model.pt"}}
```

The block is a record rather than a recipe: nothing consumes it at load time. It is
also **literal-only**, meaning every value round-trips through
`str`/`ast.literal_eval` in the saved settings. The `version` field allows the format
to evolve safely.

## Building and running a chain

The standard chains are assembled from model metadata by the `GWComposedSampler`
builders (see [](inference.md) for usage):

`from_model(model, event_data, event_metadata, fixed_context_parameters=None)`
: A single-network chain: plain NPE, or [prior conditioning](bns.md) for a model
  with `context_parameters`, with the pinned values as the chain root.

`from_gnpe_models(init_model, main_model, event_data, event_metadata, num_iterations=30)`
: Multi-iteration time GNPE: a `GibbsBlock` cycling the kernel and main-network
  factors. The chain is density-free.

`from_singlestep_gnpe(main_model, proxy_source, event_data, event_metadata)`
: Single-step, density-preserving GNPE. The `proxy_source` supplies the proxies (a
  `DeltaFactor` for prior conditioning, or an unconditional NDE for
  [density recovery](result.md#density-recovery)), followed by the main network and
  the kernel correction.

A chain is ordinary Python, and the builders have no privileged machinery:

```python
from dingo.core.factors import ChainComposer, FlowFactor
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import RAToEventFrame

context = GWSamplerContext.from_model(model, event_data, event_metadata)
flow = FlowFactor.from_model(model, aliases={"ra": "ra@t_ref"})
composer = ChainComposer([flow, RAToEventFrame()])

samples = composer.sample(10_000, context, batch_size=5_000)
log_prob = composer.log_prob(samples, context)  # the reverse fold, re-evaluated
```

This is the chain that `from_model` assembles for a plain NPE model (plus a
`DeltaFactor` filler for any delta-prior parameters). `GWComposedSampler` adds the
DataFrame runner (`run_sampler`) and the `Result` export (`to_result` / `to_hdf5`).

### Writing a new step

1. **Pick the type**: a step that samples new parameters is a `Factor`; one that
   transforms existing ones deterministically and invertibly is a
   `Reparametrization`; one that annotates the importance-sampling target is a
   `TargetCorrection`.
2. **Declare the interface**: `parameters` (the columns emitted) and `conditioning`
   (the columns read), plus `consumes` / `produces` for side-channel columns.
3. **Implement the contract.** A factor implements `sample_and_log_prob` (returning
   `num_samples` draws per conditioning row, flattened in row-major order) and
   `log_prob`, both in physical parameter space. A reparametrization implements
   `forward` and `inverse` (and `log_det` when the map is not measure-preserving);
   the inverse must rebuild exactly the consumed columns, since the reverse fold
   depends on it. A target correction implements `correction`.
4. **Read data only through the context**, so the step remains valid under a derived
   context.
5. **Override `describe()`** if the step has salient configuration, keeping the
   descriptor literal-only.

## API

```{eval-rst}
.. autoclass:: dingo.core.factors.Factor
    :members:
    :show-inheritance:

.. autoclass:: dingo.core.factors.Reparametrization
    :members:
    :show-inheritance:

.. autoclass:: dingo.core.factors.TargetCorrection
    :members:
    :show-inheritance:

.. autoclass:: dingo.core.factors.Stage
    :members:
    :show-inheritance:

.. autoclass:: dingo.core.factors.ChainComposer
    :members:
    :show-inheritance:

.. autoclass:: dingo.gw.inference.context.GWSamplerContext
    :members:
    :show-inheritance:
```
