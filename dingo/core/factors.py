"""
Factorized sampler core: the domain-agnostic spine of the factorized-sampler design
(see vault/Hackathon/Factorized_Sampler_Design.md).

The posterior is an ordered product of conditional factors,

    q(theta_1, ..., theta_n | d) = prod_i q_i(theta_i | f_i(theta_<i, d)),

where each ``Factor`` samples one parameter block and returns its own log-prob, and a
composer evaluates them. ``ChainComposer`` runs the factors autoregressively.

Factors work in physical parameter space; a network's standardized space exists only
inside its forward pass, mediated by ``Standardization``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Union, runtime_checkable

import pandas as pd
import torch

from dingo.core.posterior_models import BasePosteriorModel


class Standardization:
    """
    Affine map between a network's standardized space (``z = (theta - mean) / std``) and
    physical parameter space.

    Holds one network's ``mean`` and ``std`` and applies them in both directions:
    de-standardizing network outputs to physical samples, and standardizing physical
    parameters for ``log_prob``. Different factors (e.g. a GNPE init and main network)
    carry different instances.
    """

    def __init__(self, mean: dict[str, float], std: dict[str, float]):
        self.mean = dict(mean)
        self.std = dict(std)

    def standardize(
        self, values: dict[str, torch.Tensor], names: list[str]
    ) -> torch.Tensor:
        """Map physical ``values`` (a dict of named tensors) to a standardized tensor
        with columns in ``names`` order."""
        cols = [(values[n] - self.mean[n]) / self.std[n] for n in names]
        return torch.stack(cols, dim=-1)

    def destandardize(
        self, z: torch.Tensor, names: list[str]
    ) -> dict[str, torch.Tensor]:
        """Map a standardized tensor (columns in ``names`` order) back to a dict of
        named physical tensors."""
        return {n: z[..., i] * self.std[n] + self.mean[n] for i, n in enumerate(names)}

    def log_det(self, names: list[str]) -> float:
        """The term to *add* to a network log-prob to express it in physical space:
        ``log p_theta = log p_z - sum log std``. Same correction in both directions."""
        return -sum(math.log(self.std[n]) for n in names)


@runtime_checkable
class SamplerContext(Protocol):
    """
    Per-event shared state referenced by every factor: the data ``d`` and quantities
    derived from it (the prepared-data representation, the likelihood, event metadata).
    Concrete implementations are domain-specific; see
    ``dingo.gw.inference.factors.GWSamplerContext``.
    """

    event_metadata: Optional[dict]

    def prepared_data(self) -> torch.Tensor:
        """The data representation the factors condition on
        (whiten/decimate/repackage/...), computed once and cached."""
        ...

    def likelihood(self):
        """The likelihood, for likelihood-based factors (synthetic phase) and IS."""
        ...


@dataclass
class Conditioning:
    """
    The conditioning passed to a factor: the shared ``context``, and ``given``, the
    physical values of the earlier-block parameters the factor conditions on.
    """

    context: SamplerContext
    given: dict[str, torch.Tensor] = field(default_factory=dict)


def _cat_dict(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate a list of ``name -> tensor`` dicts along dim 0 (re-joining batches)."""
    return {k: torch.cat([b[k] for b in batches]) for k in batches[0]}


def chunk_and_concat(
    total: int,
    batch_size: Optional[int],
    run_once: Callable[[int], tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Run ``run_once`` over batches of ``total`` and concatenate the results.

    Splits ``total`` into chunks of ``batch_size``, calls ``run_once(n) -> (samples,
    log_prob)`` per chunk, and concatenates the sample dicts and log-probs. Caps peak
    memory at one chunk. ``batch_size=None`` runs ``total`` in a single call. ``log_prob``
    may be ``None`` (for a composer without a tractable density, i.e. Gibbs)."""
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


class Factor(ABC):
    """
    A conditional distribution ``q_i(theta_i | f_i(theta_<i, d))`` over one parameter
    block, emitting physical-space samples and a physical-space log-prob.

    A call draws ``num_samples`` samples per conditioning row and returns ``n_rows *
    num_samples`` rows in row-major order, where ``n_rows`` is the number of rows in
    ``cond.given`` (1 if unconditioned). This mirrors the network's
    ``sample_and_log_prob(*context, num_samples=n) -> (n_rows, n, dim)``.

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
        self, num_samples: int, cond: Conditioning
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Draw ``num_samples`` samples per conditioning row; return ``(samples,
        log_prob)`` in physical space. The samples dict may include named columns beyond
        ``parameters``."""

    @abstractmethod
    def log_prob(
        self, theta_i: dict[str, torch.Tensor], cond: Conditioning
    ) -> torch.Tensor:
        """Evaluate ``log q_i`` at given physical ``theta_i`` (one conditioning row per
        row)."""


def _base_model_metadata(model: BasePosteriorModel) -> dict:
    """The training metadata describing the parameter standardization / data settings.
    For an unconditional (density-recovery) model this lives under ``metadata["base"]``.
    """
    metadata = model.metadata
    if metadata["train_settings"]["data"].get("unconditional", False):
        return metadata["base"]
    return metadata


class FlowFactor(Factor):
    """
    Factor wrapping a posterior model (NPE flow, FMPE, ...). Reaches data only through
    ``SamplerContext.prepared_data()`` and encapsulates the network's standardization, so
    its interface is in physical parameter space.

    An unconditioned model draws from the shared data context; a model with
    ``context_parameters`` conditions on the values in ``cond.given``.
    """

    def __init__(
        self,
        model: BasePosteriorModel,
        parameters: list[str],
        conditioning: Optional[list[str]] = None,
        context_parameters: Optional[list[str]] = None,
    ):
        self.model = model
        self.parameters = parameters
        self.conditioning = conditioning or []
        # Network conditioning inputs (GNPE proxies). Empty for plain NPE.
        self.context_parameters = context_parameters or []
        std = _base_model_metadata(model)["train_settings"]["data"]["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @classmethod
    def from_model(cls, model: BasePosteriorModel) -> "FlowFactor":
        """Build a factor from a model, reading ``parameters`` and ``context_parameters``
        from its training metadata."""
        data_settings = _base_model_metadata(model)["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        return cls(
            model=model,
            parameters=data_settings["inference_parameters"],
            conditioning=list(context_parameters),
            context_parameters=list(context_parameters),
        )

    def sample_and_log_prob(self, num_samples, cond):
        """Draw ``num_samples`` samples per conditioning row. Unconditioned: ``num_samples``
        draws from the shared data context. Conditioned: ``num_samples`` per context row,
        returning ``n_rows * num_samples`` rows in row-major order."""
        data = cond.context.prepared_data()
        self.model.network.eval()
        if not self.context_parameters:
            with torch.no_grad():
                z, log_prob = self.model.sample_and_log_prob(
                    data.unsqueeze(0), num_samples=num_samples
                )
            # Squeeze the batch dimension added for the single shared context.
            z = z.squeeze(0)
            log_prob = log_prob.squeeze(0)
        else:
            # Standardize the (physical) conditioning the network was trained on, giving
            # B = N context rows; expand the shared data to match. The network draws
            # num_samples per row -> (N, num_samples, dim).
            ctx = self.standardization.standardize(cond.given, self.context_parameters)
            n_rows = ctx.shape[0]
            data = data.expand(n_rows, *data.shape)
            with torch.no_grad():
                z, log_prob = self.model.sample_and_log_prob(
                    data, ctx, num_samples=num_samples
                )
            z = z.reshape(n_rows * num_samples, z.shape[-1])
            log_prob = log_prob.reshape(n_rows * num_samples)
        theta = self.standardization.destandardize(z, self.parameters)
        log_prob = log_prob + self.standardization.log_det(self.parameters)
        return theta, log_prob

    def log_prob(self, theta_i, cond):
        num_samples = next(iter(theta_i.values())).shape[0]
        z = self.standardization.standardize(theta_i, self.parameters)
        data = cond.context.prepared_data()
        data = data.expand(num_samples, *data.shape)
        net_context: tuple[torch.Tensor, ...]
        if not self.context_parameters:
            net_context = (data,)
        else:
            ctx = self.standardization.standardize(cond.given, self.context_parameters)
            net_context = (data, ctx)
        self.model.network.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(z, *net_context)
        return log_prob + self.standardization.log_det(self.parameters)


@dataclass
class Stage:
    """A chain factor and its fan-out: the number of samples drawn per incoming
    conditioning row. The root stage draws the base count and ignores ``fan_out``."""

    factor: Factor
    fan_out: int = 1


class ChainComposer:
    """
    Autoregressive composer over an ordered list of ``Stage``s.

    Draws the stages in declared order -- a topological order of the conditioning DAG --
    expanding each stage by its fan-out and summing the log-probs. Covers plain NPE,
    single-step GNPE, prior conditioning, synthetic phase, and intrinsic/extrinsic splits.
    Multi-iteration (cyclic) GNPE uses ``GibbsComposer`` instead.

    Accepts bare factors (wrapped as ``Stage(factor, fan_out=1)``) or explicit ``Stage``s.
    """

    def __init__(self, stages: list[Union["Stage", Factor]]):
        self.stages = [s if isinstance(s, Stage) else Stage(s) for s in stages]
        self._validate()

    def _validate(self):
        """Check the declared order is a valid topological order: every conditioning
        name is produced by an earlier factor."""
        produced: set[str] = set()
        for factor in self.factors:
            missing = [c for c in factor.conditioning if c not in produced]
            if missing:
                raise ValueError(
                    f"Factor producing {factor.parameters} conditions on {missing}, "
                    f"which no earlier factor produces. Check chain order."
                )
            produced.update(factor.parameters)

    @property
    def factors(self) -> list[Factor]:
        """The stage factors, in order."""
        return [stage.factor for stage in self.stages]

    @property
    def expansion(self) -> int:
        """Product of the non-root fan-outs; ``sample`` returns ``num_samples *
        expansion`` rows."""
        return math.prod(stage.fan_out for stage in self.stages[1:])

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Draw samples. ``num_samples`` is the base (root) count; the result has
        ``num_samples * expansion`` rows. ``batch_size`` chunks the base count (``None``
        draws in one pass)."""
        return chunk_and_concat(
            num_samples, batch_size, lambda n: self._run_chain_once(n, context)
        )

    def _run_chain_once(
        self, base: int, context: SamplerContext
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """One pass of the whole chain for ``base`` root samples."""
        samples: dict[str, torch.Tensor] = {}
        total: torch.Tensor | float = 0.0
        for i, stage in enumerate(self.stages):
            factor = stage.factor
            n = base if i == 0 else stage.fan_out
            cond = Conditioning(context, {k: samples[k] for k in factor.conditioning})
            block, lp = factor.sample_and_log_prob(n, cond)
            # The factor returned (rows_so_far * n) rows. Expand the carried columns to
            # match -- each upstream row repeated n times (the block is flattened in the
            # same row-major order). The root has no carried rows, and fan_out=1 stages
            # are a no-op, so 1:1 chains are untouched.
            if i > 0 and n > 1:
                samples = {k: v.repeat_interleave(n, 0) for k, v in samples.items()}
                if torch.is_tensor(total):
                    total = total.repeat_interleave(n, 0)
            samples.update(block)
            total = total + lp
        return samples, total

    def log_prob(
        self, samples: dict[str, torch.Tensor], context: SamplerContext
    ) -> torch.Tensor:
        total: torch.Tensor | float = 0.0
        for factor in self.factors:
            cond = Conditioning(context, {k: samples[k] for k in factor.conditioning})
            theta_i = {k: samples[k] for k in factor.parameters}
            total = total + factor.log_prob(theta_i, cond)
        return total

    def sample(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Per-sample dict of parameters plus ``log_prob``."""
        samples, log_prob = self.sample_and_log_prob(num_samples, context, batch_size)
        return {**samples, "log_prob": log_prob}


class GibbsComposer:
    """
    Blocked Gibbs composer over a cyclic conditioning dependency.

    Seeds the chain with an init factor, then sweeps the factor list in order for
    ``num_iterations`` iterations; each factor conditions on the current state and
    overwrites its own block. The cyclic dependency has no tractable marginal, so
    ``sample`` returns parameters without a ``log_prob`` (recoverable by fitting an
    unconditional density to the samples and taking one ``ChainComposer`` step). Dingo uses
    this only for multi-iteration GNPE (the GNPE factors in ``dingo.gw.inference.factors``),
    but the loop is generic.

    Batching is vertical, via ``chunk_and_concat``: chunk the walkers and run the whole
    loop per chunk.
    """

    def __init__(self, init_factor: Factor, factors: list[Factor], num_iterations: int):
        self.init_factor = init_factor
        self.factors = list(factors)
        self.num_iterations = num_iterations

    def sample(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        samples, _ = chunk_and_concat(
            num_samples, batch_size, lambda n: (self._run_once(n, context), None)
        )
        return samples

    def _run_once(
        self, num_samples: int, context: SamplerContext
    ) -> dict[str, torch.Tensor]:
        # Seed the chain (e.g. an init network's detector times); the walkers are the rows.
        seed, _ = self.init_factor.sample_and_log_prob(
            num_samples, Conditioning(context)
        )
        state = dict(seed)
        for _ in range(self.num_iterations):
            for factor in self.factors:
                cond = Conditioning(context, {k: state[k] for k in factor.conditioning})
                # One sample per walker (Gibbs is 1:1); walkers are the conditioning rows.
                block, _ = factor.sample_and_log_prob(1, cond)
                state.update(block)
        # Each factor's parameter block (e.g. proxies + inference parameters), dropping
        # side-channel columns such as the recomputed detector times.
        return {p: state[p] for factor in self.factors for p in factor.parameters}


class Composer(Protocol):
    """Interface required by ``ComposedSampler``: ``sample`` returns a per-sample dict of
    parameters (plus ``log_prob`` for density-preserving composers). Satisfied by
    ``ChainComposer`` and ``GibbsComposer``."""

    def sample(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]: ...


class ComposedSampler:
    """
    Façade over a ``Composer`` and a ``SamplerContext``. Runs the composer, applies
    domain-specific post-processing, and returns the samples as a DataFrame.
    """

    def __init__(self, composer: Composer, context: SamplerContext):
        self.composer = composer
        self.context = context
        self.samples: Optional[pd.DataFrame] = None

    def _post_process(self, samples: dict, inverse: bool = False):
        """Hook for domain-specific post-processing; no-op by default."""
        pass

    def run_sampler(
        self, num_samples: int, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Draw ``num_samples`` samples (chunked by ``batch_size``), post-process, and
        return them as a DataFrame. ``ChainComposer`` includes ``log_prob``;
        ``GibbsComposer`` does not."""
        merged = self.composer.sample(num_samples, self.context, batch_size)
        merged = {k: v.cpu().numpy() for k, v in merged.items()}
        self._post_process(merged)
        self.samples = pd.DataFrame(merged)
        return self.samples
