"""
Factorized sampler core: the domain-agnostic spine of the factorized-sampler design
(see vault/Hackathon/Factorized_Sampler_Design.md).

The posterior is an ordered product of conditional factors,

    q(theta_1, ..., theta_n | d) = prod_i q_i(theta_i | f_i(theta_<i, d)),

where each `Factor` samples one parameter block and returns its own log-prob, and a
composer evaluates them. `ChainComposer` runs the factors autoregressively.

Factors work in physical parameter space; a network's standardized space exists only
inside its forward pass, mediated by `Standardization`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union, runtime_checkable

import pandas as pd
import torch

from dingo.core.posterior_models import BasePosteriorModel


class Standardization:
    """
    Affine map between a network's standardized space (`z = (theta - mean) / std`) and
    physical parameter space.

    Holds one network's `mean` and `std` and applies them in both directions:
    de-standardizing network outputs to physical samples, and standardizing physical
    parameters for `log_prob`. Different factors (e.g. a GNPE init and main network)
    carry different instances.
    """

    def __init__(self, mean: dict[str, float], std: dict[str, float]):
        self.mean = dict(mean)
        self.std = dict(std)

    def standardize(
        self, values: dict[str, torch.Tensor], names: list[str]
    ) -> torch.Tensor:
        """Map physical `values` (a dict of named tensors) to a standardized tensor
        with columns in `names` order."""
        cols = [(values[n] - self.mean[n]) / self.std[n] for n in names]
        return torch.stack(cols, dim=-1)

    def destandardize(
        self, z: torch.Tensor, names: list[str]
    ) -> dict[str, torch.Tensor]:
        """Map a standardized tensor (columns in `names` order) back to a dict of
        named physical tensors."""
        return {n: z[..., i] * self.std[n] + self.mean[n] for i, n in enumerate(names)}

    def log_det(self, names: list[str]) -> float:
        """The term to *add* to a network log-prob to express it in physical space:
        `log p_theta = log p_z - sum log std`. Same correction in both directions."""
        return -sum(math.log(self.std[n]) for n in names)


@runtime_checkable
class SamplerContext(Protocol):
    """
    Per-event shared state referenced by every factor: the data `d` and quantities
    derived from it (the prepared-data representation, the likelihood, event metadata).
    Concrete implementations are domain-specific; see
    `dingo.gw.inference.factors.GWSamplerContext`.

    `device` is the torch device the chain runs on: steps that create fresh tensors
    (rather than transforming existing ones) create them here, so their outputs can
    join a chain whose network factors run on a GPU.
    """

    event_metadata: Optional[dict]
    device: Union[torch.device, str]

    def prepared_data(self) -> torch.Tensor:
        """The data representation the factors condition on
        (whiten/decimate/repackage/...), computed once and cached."""
        ...

    def likelihood(self):
        """The likelihood, for likelihood-based factors (synthetic phase) and IS."""
        ...


def _cat_dict(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate a list of `name -> tensor` dicts along dim 0 (re-joining batches)."""
    return {k: torch.cat([b[k] for b in batches]) for k in batches[0]}


def chunk_and_concat(
    total: int,
    batch_size: Optional[int],
    run_once: Callable[[int], tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Run `run_once` over batches of `total` and concatenate the results.

    Caps peak memory at one chunk. `log_prob` may be `None` (for a composer without a
    tractable density, i.e. Gibbs).

    Parameters
    ----------
    total : int
        Total number of samples to produce.
    batch_size : int, optional
        Chunk size; `None` runs `total` in a single call.
    run_once : callable
        `run_once(n) -> (samples, log_prob)` for one chunk of `n` samples.

    Returns
    -------
    samples : dict[str, torch.Tensor]
        The concatenated per-name sample tensors.
    log_prob : torch.Tensor or None
        The concatenated log-probs, or `None` when `run_once` returns `None`.
    """
    bs = batch_size or total
    sample_parts: list[dict[str, torch.Tensor]] = []
    lp_parts: list[Optional[torch.Tensor]] = []
    for start in range(0, total, bs):
        block, lp = run_once(min(bs, total - start))
        sample_parts.append(block)
        lp_parts.append(lp)
    samples = _cat_dict(sample_parts)
    log_prob = None if lp_parts[0] is None else torch.cat(lp_parts)
    return samples, log_prob


def _describe_default(step) -> dict:
    """Default provenance descriptor for a chain step: class name, the parameter
    block it emits, and what it conditions on. Literal-only (round-trips through
    `str`/`ast.literal_eval` in saved settings)."""
    return {
        "step": type(step).__name__,
        "parameters": list(step.parameters),
        "conditioning": list(step.conditioning),
    }


class Factor(ABC):
    """
    A conditional distribution `q_i(theta_i | f_i(theta_<i, d))` over one parameter
    block, emitting physical-space samples and a physical-space log-prob.

    A call draws `num_samples` samples per conditioning row and returns `n_rows *
    num_samples` rows in row-major order, where `n_rows` is the number of rows in
    `given` (1 if unconditioned). This mirrors the network's
    `sample_and_log_prob(*context, num_samples=n) -> (n_rows, n, dim)`.

    Attributes
    ----------
    parameters : list[str]
        The parameter block this factor produces.
    conditioning : list[str]
        Earlier-block parameters it conditions on (data dependence is via the context).
    """

    parameters: list[str]
    conditioning: list[str]

    @abstractmethod
    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Draw `num_samples` samples per conditioning row; return `(samples,
        log_prob)` in physical space. The samples dict may include named columns beyond
        `parameters`."""

    @abstractmethod
    def log_prob(
        self,
        theta_i: dict[str, torch.Tensor],
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Evaluate `log q_i` at given physical `theta_i` (one conditioning row per
        row)."""

    def describe(self) -> dict:
        """Provenance descriptor for saved-result metadata; steps with salient
        configuration override this."""
        return _describe_default(self)


def _base_model_metadata(model: BasePosteriorModel) -> dict:
    """The base *analysis* metadata (dataset / domain / detector / data settings): for
    an unconditional (density-recovery) model this lives under `metadata["base"]`, the
    metadata of the model whose samples it was trained on. The network-bound settings
    (`standardization`, `inference_parameters`) are always the model's own and are read
    from `model.metadata` directly."""
    metadata = model.metadata
    if metadata["train_settings"]["data"].get("unconditional", False):
        return metadata["base"]
    return metadata


class FlowFactor(Factor):
    """
    Factor wrapping a posterior model (NPE flow, FMPE, ...). Encapsulates the network's
    standardization, so its interface is in physical parameter space.

    Three conditioning shapes: a data-conditional model draws from the shared
    `SamplerContext.prepared_data()`; a model with `context_parameters` (GNPE proxies)
    additionally conditions on the values in `given`; and an *unconditional* model
    (`unconditional` in its training metadata -- a density-recovery NDE or a
    model-as-prior) takes no input at all and never touches the context.
    """

    def __init__(
        self,
        model: BasePosteriorModel,
        parameters: list[str],
        conditioning: Optional[list[str]] = None,
        context_parameters: Optional[list[str]] = None,
        aliases: Optional[dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        model : BasePosteriorModel
            The posterior model (NPE flow, FMPE, ...) wrapped by this factor.
        parameters : list[str]
            The network's trained parameter names (standardization is keyed by these).
        conditioning : list[str], optional
            Earlier-block parameters this factor conditions on.
        context_parameters : list[str], optional
            Network conditioning inputs (GNPE proxies); empty for plain NPE.
        aliases : dict[str, str], optional
            Trained-name to exposed-name map at the factor boundary (e.g.
            `{"ra": "ra@t_ref"}`), so a downstream reparametrization can convert frames
            by name without retraining.
        """
        self.model = model
        # The network's trained parameter names -- standardization is keyed by these. The
        # factor exposes them under canonical aliases (e.g. ra -> ra@t_ref), so a downstream
        # reparametrization can convert frames by name without retraining (design Q#7).
        self._net_parameters = parameters
        self.aliases = aliases or {}
        self.parameters = [self.aliases.get(p, p) for p in parameters]
        self.conditioning = conditioning or []
        # Network conditioning inputs (GNPE proxies). Empty for plain NPE.
        self.context_parameters = context_parameters or []
        data_settings = model.metadata["train_settings"]["data"]
        self.unconditional = data_settings.get("unconditional", False)
        # The model's *own* standardization: an unconditional (density-recovery) NDE
        # carries its own, distinct from the base model's under metadata["base"].
        std = data_settings["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @classmethod
    def from_model(
        cls, model: BasePosteriorModel, aliases: Optional[dict[str, str]] = None
    ) -> "FlowFactor":
        """Build a factor from a model, reading `parameters` and `context_parameters`
        from its own training metadata (for an unconditional NDE these are its own,
        e.g. the GNPE proxies it was trained on).

        Parameters
        ----------
        model : BasePosteriorModel
            The posterior model to wrap.
        aliases : dict[str, str], optional
            Trained-name to exposed canonical-name map (e.g. `{"ra": "ra@t_ref"}`) at
            the factor boundary.

        Returns
        -------
        FlowFactor
        """
        data_settings = model.metadata["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        return cls(
            model=model,
            parameters=data_settings["inference_parameters"],
            conditioning=list(context_parameters),
            context_parameters=list(context_parameters),
            aliases=aliases,
        )

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw `num_samples` samples per conditioning row. Unconditional: draws with
        no input (the context is not touched). Data-conditional: `num_samples` draws
        from the shared data context. Parameter-conditioned: `num_samples` per context
        row, returning `n_rows * num_samples` rows in row-major order."""
        self.model.network.eval()
        if self.unconditional:
            with torch.no_grad():
                z, log_prob = self.model.sample_and_log_prob(num_samples=num_samples)
        elif not self.context_parameters:
            data = context.prepared_data()
            with torch.no_grad():
                z, log_prob = self.model.sample_and_log_prob(
                    data.unsqueeze(0), num_samples=num_samples
                )
            # Squeeze the batch dimension added for the single shared context.
            z = z.squeeze(0)
            log_prob = log_prob.squeeze(0)
        else:
            data = context.prepared_data()
            # Standardize the (physical) conditioning the network was trained on, giving
            # B = N context rows; expand the shared data to match. The network draws
            # num_samples per row -> (N, num_samples, dim).
            # TODO: the embedding runs once per row, so row-identical data with
            # row-varying conditioning (the intrinsic/extrinsic split) recomputes N
            # identical data embeddings. Fixing this needs an embed/fuse split on the
            # model (FlowWrapper) so the cached embedding can be reused across rows.
            ctx = self.standardization.standardize(given, self.context_parameters)
            n_rows = ctx.shape[0]
            data = data.expand(n_rows, *data.shape)
            with torch.no_grad():
                z, log_prob = self.model.sample_and_log_prob(
                    data, ctx, num_samples=num_samples
                )
            z = z.reshape(n_rows * num_samples, z.shape[-1])
            log_prob = log_prob.reshape(n_rows * num_samples)
        theta = self.standardization.destandardize(z, self._net_parameters)
        log_prob = log_prob + self.standardization.log_det(self._net_parameters)
        # Expose trained names under their canonical aliases at the factor boundary.
        theta = {self.aliases.get(k, k): v for k, v in theta.items()}
        return theta, log_prob

    def log_prob(self, theta_i, context, given=None):
        # theta_i uses exposed (aliased) names; map back to the network's trained names.
        theta_net = {
            net: theta_i[self.aliases.get(net, net)] for net in self._net_parameters
        }
        num_samples = next(iter(theta_net.values())).shape[0]
        z = self.standardization.standardize(theta_net, self._net_parameters)
        net_context: tuple[torch.Tensor, ...]
        if self.unconditional:
            net_context = ()
        else:
            data = context.prepared_data()
            data = data.expand(num_samples, *data.shape)
            if not self.context_parameters:
                net_context = (data,)
            else:
                ctx = self.standardization.standardize(given, self.context_parameters)
                net_context = (data, ctx)
        self.model.network.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(z, *net_context)
        return log_prob + self.standardization.log_det(self._net_parameters)

    def describe(self) -> dict:
        return {**_describe_default(self), "unconditional": self.unconditional}


class DeltaFactor(Factor):
    """`q_i = delta(theta_i - c)`: a point mass pinning parameters to fixed values,
    contributing 0 to the proposal log-prob.

    Used as the chain root for prior-conditioning or known proxies (where later factors
    condition on the pinned values), and as a non-root filler for delta-prior parameters
    a model does not infer (one constant per current row).
    """

    def __init__(self, values: dict[str, float]):
        self.values = values
        self.parameters = list(values)
        self.conditioning = []

    def sample_and_log_prob(self, num_samples, context, given=None):
        # A delta factor creates fresh tensors, so it places them on the chain's
        # device (unlike steps that transform existing rows, which follow their
        # inputs).
        device = getattr(context, "device", None)
        samples = {
            p: torch.full((num_samples,), float(v), device=device)
            for p, v in self.values.items()
        }
        return samples, torch.zeros(num_samples, device=device)

    def log_prob(self, theta_i, context, given=None):
        """0 per row, matching sample time: the point mass is evaluated on its own
        support (the chain only re-plugs its own samples), so the pinned block is
        conditioned on, not integrated over. Off-support densities are not
        represented."""
        # One zero per row, on the same device/dtype as the evaluated block.
        reference_column = next(iter(theta_i.values()))
        return torch.zeros_like(reference_column)

    def describe(self) -> dict:
        return {
            **_describe_default(self),
            "values": {k: float(v) for k, v in self.values.items()},
        }


class SampleTableFactor(Factor):
    """
    Chain root emitting a fixed table of existing samples (with their stored proposal
    log-prob, if available) instead of drawing new ones.

    This is how a chain continues from previously drawn samples: the
    importance-sampling side runs its post-sampling steps (e.g. synthetic phase) as a
    chain rooted in the proposal sample table, and the composer's ordinary log-prob
    fold then yields the joint proposal density `log q(theta) + log q(extra | theta)`
    with no special-casing. Without a stored log-prob the chain is density-free (as
    with a `GibbsBlock`).
    """

    def __init__(self, table: dict, log_prob=None):
        """
        Parameters
        ----------
        table : dict
            The existing samples, one array-like entry per parameter column.
        log_prob : array-like, optional
            The stored proposal log-prob of the table rows. If omitted, the chain
            has no tractable density.
        """
        self.table = {k: torch.as_tensor(v) for k, v in table.items()}
        self.table_log_prob = (
            torch.as_tensor(log_prob) if log_prob is not None else None
        )
        self.parameters = list(self.table)
        self.conditioning: list[str] = []

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Emit the table; `num_samples` must equal the table length (a fixed table
        cannot be chunked, so run the chain with `batch_size=None`)."""
        n = len(next(iter(self.table.values())))
        if num_samples != n:
            raise ValueError(
                f"A sample table is fixed: num_samples must equal the table length "
                f"({n}), got {num_samples}. Run the chain with batch_size=None."
            )
        return dict(self.table), self.table_log_prob

    def log_prob(self, theta_i, context, given=None):
        raise NotImplementedError(
            "A sample table is not a density; its rows carry their stored log-prob. "
            "Evaluate log_prob through the chain that produced the samples instead."
        )


class Reparametrization(ABC):
    """
    A deterministic bijection `Step`: it transforms existing parameters (no sampling)
    and contributes `-log|det J|` to the proposal density.

    Unlike a `Factor` it is 1:1 and invertible -- `forward` maps the conditioning block
    to the `parameters` block, `inverse` maps back (for re-plug / importance sampling),
    and its density contribution is a Jacobian, not a sampled log-prob. Used to relate a
    network's coordinates to physical ones (e.g. right ascension from the training reference
    frame to the event frame). Subclasses implement `forward` / `inverse` and, where the
    map is not measure-preserving, `log_det`.
    """

    parameters: list[str]
    conditioning: list[str]

    @abstractmethod
    def forward(
        self, given: dict[str, torch.Tensor], context: "SamplerContext"
    ) -> dict[str, torch.Tensor]:
        """Map the conditioning block to the `parameters` block."""

    @abstractmethod
    def inverse(
        self,
        params: dict[str, torch.Tensor],
        context: "SamplerContext",
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Rebuild the consumed inputs from the `parameters` block. `given` holds
        the non-consumed conditioning columns still present in the chain (e.g. a
        proxy the bijection shifts by, or invariant parameters a coordinate
        change needs); bijections that depend only on their own outputs may
        ignore it."""

    def log_det(
        self, given: dict[str, torch.Tensor], context: "SamplerContext"
    ) -> torch.Tensor:
        """`log|det J|` of `forward`, per row. Default 0 (measure-preserving), on
        the device of the transformed rows."""
        reference_column = next(iter(given.values()))
        return torch.zeros_like(reference_column)

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Apply `forward` to the conditioning; contribute `-log|det J|`. `num_samples`
        must be 1 (a reparametrization is 1:1)."""
        if num_samples != 1:
            raise ValueError("A reparametrization is 1:1; use fan_out=1.")
        out = self.forward(given, context)
        return out, -self.log_det(given, context)

    @property
    def consumes(self) -> list[str]:
        """Inputs the bijection replaces with its outputs (dropped after the step).
        `ChainComposer.log_prob` rebuilds them via `inverse`."""
        return [c for c in self.conditioning if c not in self.parameters]

    def describe(self) -> dict:
        """Provenance descriptor for saved-result metadata."""
        return _describe_default(self)


class TargetCorrection(ABC):
    """
    A `Step` that emits an importance-sampling target correction as a side-channel column
    and contributes nothing to the proposal density.

    Its value belongs to the IS target, not the proposal: it reads earlier blocks, emits a
    named column, optionally consumes intermediates (`consumes`), and adds 0 to the
    proposal log-prob. 1:1.

    `consumes` must name only side-channel intermediates (e.g. recomputed detector
    times), never a sampled parameter: unlike a `Reparametrization`, a correction has no
    inverse, so anything it consumes cannot be rebuilt by `ChainComposer.log_prob`.
    """

    parameters: list[str]
    conditioning: list[str]
    consumes: list[str]

    @abstractmethod
    def correction(
        self, given: dict[str, torch.Tensor], context: "SamplerContext"
    ) -> dict[str, torch.Tensor]:
        """The side-channel column(s), one value per conditioning row."""

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if num_samples != 1:
            raise ValueError("A target correction is 1:1; use fan_out=1.")
        out = self.correction(given, context)
        # 0 proposal contribution per row, on the device of the emitted column.
        reference_column = next(iter(out.values()))
        return out, torch.zeros_like(reference_column)

    def describe(self) -> dict:
        """Provenance descriptor for saved-result metadata."""
        return _describe_default(self)


class Step(Protocol):
    """
    One entry a `ChainComposer` folds over: it emits a parameter block and an optional
    contribution to the proposal `log_prob`.

    Density-contributing steps (`Factor`) return a tensor; density-free sampling blocks
    (`GibbsBlock`) return `None`. `parameters` names the block produced, `conditioning`
    the earlier-block parameters read from the chain (data is implicit via the context).
    """

    parameters: list[str]
    conditioning: list[str]

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]: ...


@dataclass
class Stage:
    """A chain `Step` and its fan-out: the number of samples drawn per incoming
    conditioning row. The root stage draws the base count and ignores `fan_out`."""

    step: Step
    fan_out: int = 1


class ChainComposer:
    """
    Autoregressive composer over an ordered list of `Stage`s.

    Folds the steps in declared order -- a topological order of the conditioning DAG --
    expanding each by its fan-out and summing the proposal log-probs. A step is a
    `Factor` (contributes `log q_i`) or a density-free sampling block (`GibbsBlock`,
    contributes `None`); if any step is density-free the chain has no tractable density
    and `sample` omits `log_prob`. Covers plain NPE, single-step GNPE, prior
    conditioning, synthetic phase, intrinsic/extrinsic splits, and -- via `GibbsBlock` --
    multi-iteration GNPE.

    Accepts bare steps (wrapped as `Stage(step, fan_out=1)`) or explicit `Stage`s.
    """

    def __init__(self, stages: list[Union["Stage", Step]]):
        self.stages = [s if isinstance(s, Stage) else Stage(s) for s in stages]
        self._validate()

    def _validate(self):
        """Check the declared order is a valid topological order: every conditioning
        name is produced by an earlier step, and no step overwrites an existing
        column -- except a `Reparametrization` replacing its own inputs, which is
        invertible, so `log_prob` can restore the overwritten state. A step's
        emitted columns default to its `parameters`, but a step may emit
        side-channel columns too (`produces`)."""
        produced: set[str] = set()
        for step in self.steps:
            missing = [c for c in step.conditioning if c not in produced]
            if missing:
                raise ValueError(
                    f"A step producing {step.parameters} conditions on {missing}, "
                    f"which no earlier step produces. Check chain order."
                )
            emitted = set(getattr(step, "produces", step.parameters))
            replaceable = (
                set(step.conditioning) if isinstance(step, Reparametrization) else set()
            )
            clobbered = (emitted & produced) - replaceable
            if clobbered:
                raise ValueError(
                    f"A step producing {step.parameters} would overwrite existing "
                    f"column(s) {sorted(clobbered)}. Only a Reparametrization may "
                    f"replace columns (its inverse can rebuild them for log_prob)."
                )
            produced.update(emitted)

    @property
    def steps(self) -> list[Step]:
        """The stage steps, in order."""
        return [stage.step for stage in self.stages]

    @property
    def expansion(self) -> int:
        """Product of the non-root fan-outs; `sample` returns `num_samples *
        expansion` rows."""
        return math.prod(stage.fan_out for stage in self.stages[1:])

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Draw samples. `num_samples` is the base (root) count; the result has
        `num_samples * expansion` rows. `batch_size` chunks the base count (`None`
        draws in one pass). The log-prob is `None` if any step is density-free."""
        return chunk_and_concat(
            num_samples, batch_size, lambda n: self._run_chain_once(n, context)
        )

    def _run_chain_once(
        self, base: int, context: SamplerContext
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """One pass of the whole chain for `base` root samples. Returns the samples and
        the summed proposal log-prob, or `None` if any step is density-free."""
        samples: dict[str, torch.Tensor] = {}
        total: torch.Tensor | float = 0.0
        has_density = True
        for i, stage in enumerate(self.stages):
            step = stage.step
            if i == 0:
                n = base  # root: draw the base count
            elif step.conditioning:
                n = stage.fan_out  # fan_out samples per conditioning row
            else:
                # Unconditioned non-root step (e.g. a fixed/delta factor): draw one value
                # per current row -- it fills the batch rather than fanning out.
                n = len(next(iter(samples.values())))
            given = {k: samples[k] for k in step.conditioning}
            block, lp = step.sample_and_log_prob(n, context, given)
            # A conditioned fan-out (fan_out > 1) expands the batch: repeat each carried row
            # to align with the step's fan_out sub-rows (the block is flattened row-major).
            # 1:1 stages and unconditioned fillers leave the batch untouched.
            if i > 0 and step.conditioning and stage.fan_out > 1:
                fan = stage.fan_out
                samples = {k: v.repeat_interleave(fan, 0) for k, v in samples.items()}
                if torch.is_tensor(total):
                    total = total.repeat_interleave(fan, 0)
            samples.update(block)
            # Drop any intermediates the step consumed: a reparametrization replaces its
            # inputs (in-place bijection); a target correction drops the columns it read.
            for k in getattr(step, "consumes", ()):
                samples.pop(k, None)
            # A single density-free step (Gibbs) makes the whole chain density-free.
            if lp is None:
                has_density = False
            elif has_density:
                total = total + lp
        return samples, (total if has_density else None)

    def log_prob(
        self, samples: dict[str, torch.Tensor], context: SamplerContext
    ) -> torch.Tensor:
        """Evaluate the chain's proposal log-density at given physical samples
        (re-plug / importance sampling).

        The steps are folded in exact reverse chain order, so the columns are
        restored to the state each step saw during sampling: a `Reparametrization`
        rebuilds its inputs via `inverse` (e.g. `ra@t_ref` from the event-frame
        `ra`) and contributes `-log|det J|`, a `Factor` contributes its `log_prob`,
        and a `TargetCorrection` contributes nothing (target-side only). Raises for
        a density-free chain (one containing a `GibbsBlock`).

        Parameters
        ----------
        samples : dict[str, torch.Tensor]
            The chain's emitted columns (one value per row). Consumed intermediates
            are rebuilt via the reparametrization inverses and need not be present.
        context : SamplerContext
            Per-event shared state.

        Returns
        -------
        torch.Tensor
            The proposal log-density per row.
        """
        if any(isinstance(step, GibbsBlock) for step in self.steps):
            raise ValueError(
                "The chain contains a density-free step (GibbsBlock), so its "
                "log_prob is unavailable; recover a density first (fit an "
                "unconditional model and take a chain step)."
            )
        values = dict(samples)
        total: torch.Tensor | float = 0.0
        for step in reversed(self.steps):
            if isinstance(step, TargetCorrection):
                continue
            if isinstance(step, Reparametrization):
                params = {k: values[k] for k in step.parameters}
                # The non-consumed conditioning is still present and may be
                # needed to invert (e.g. a proxy the bijection shifts by).
                available = {k: values[k] for k in step.conditioning if k in values}
                values.update(step.inverse(params, context, available))
                given = {k: values[k] for k in step.conditioning}
                total = total - step.log_det(given, context)
            else:
                given = {k: values[k] for k in step.conditioning}
                theta_i = {k: values[k] for k in step.parameters}
                total = total + step.log_prob(theta_i, context, given)
        return total

    def sample(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Per-sample dict of parameters, plus `log_prob` for an all-density chain."""
        samples, log_prob = self.sample_and_log_prob(num_samples, context, batch_size)
        if log_prob is None:
            return dict(samples)
        return {**samples, "log_prob": log_prob}


class GibbsBlock:
    """
    A density-free sampling-block `Step`: runs blocked Gibbs internally and yields no
    proposal log-prob.

    Seeds the chain with an init factor, then sweeps the factor list in order for
    `num_iterations` iterations; each factor conditions on the current state and
    overwrites its own block. As a chain `Step` it produces the swept parameter blocks
    and returns `None` for the log-prob -- the cyclic dependency has no tractable marginal
    (recoverable by fitting an unconditional density to the samples and taking one
    `ChainComposer` step). Dingo uses this only for multi-iteration GNPE (the GNPE factors
    in `dingo.gw.inference.factors`), but the loop is generic.

    Batching is handled by the enclosing `ChainComposer`: it chunks the walkers and runs
    the whole loop per chunk (`chunk_and_concat`).
    """

    def __init__(self, init_factor: Factor, factors: list[Factor], num_iterations: int):
        """
        Parameters
        ----------
        init_factor : Factor
            Seeds the chain (e.g. an init network's detector times).
        factors : list[Factor]
            The factors swept in order each iteration; each conditions on the current
            state and overwrites its own block.
        num_iterations : int
            Number of Gibbs sweeps.
        """
        self.init_factor = init_factor
        self.factors = list(factors)
        self.num_iterations = num_iterations
        # The blocks this step produces (proxies + inference parameters), dropping
        # side-channel columns such as the recomputed detector times.
        self.parameters = [p for factor in self.factors for p in factor.parameters]
        self.conditioning: list[str] = []

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], None]:
        """Run the Gibbs loop for `num_samples` walkers; return `(samples, None)`.
        `num_samples` is the walker (root) count -- Gibbs does not fan out."""
        return self._run_once(num_samples, context), None

    def describe(self) -> dict:
        """Provenance descriptor: the Gibbs structure, with nested descriptors for
        the init factor and the swept factors."""
        return {
            "step": type(self).__name__,
            "num_iterations": self.num_iterations,
            "init": self.init_factor.describe(),
            "factors": [factor.describe() for factor in self.factors],
        }

    def _run_once(
        self, num_samples: int, context: SamplerContext
    ) -> dict[str, torch.Tensor]:
        # Seed the chain (e.g. an init network's detector times); the walkers are the rows.
        seed, _ = self.init_factor.sample_and_log_prob(num_samples, context)
        state = dict(seed)
        for _ in range(self.num_iterations):
            for factor in self.factors:
                given = {k: state[k] for k in factor.conditioning}
                # One sample per walker (Gibbs is 1:1); walkers are the conditioning rows.
                block, _ = factor.sample_and_log_prob(1, context, given)
                state.update(block)
        return {p: state[p] for p in self.parameters}


class ComposedSampler:
    """
    Generic runner over a `ChainComposer` and a `SamplerContext`: draws samples and
    returns them as a DataFrame. Domain-specific processing lives in the chain's steps, so
    the runner is domain-agnostic.
    """

    def __init__(self, composer: ChainComposer, context: SamplerContext):
        self.composer = composer
        self.context = context
        self.samples: Optional[pd.DataFrame] = None

    def run_sampler(
        self, num_samples: int, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Draw `num_samples` samples (chunked by `batch_size`) and return them as a
        DataFrame. An all-density chain includes `log_prob`; a chain with a `GibbsBlock`
        step does not."""
        merged = self.composer.sample(num_samples, self.context, batch_size)
        merged = {k: v.cpu().numpy() for k, v in merged.items()}
        self.samples = pd.DataFrame(merged)
        return self.samples
