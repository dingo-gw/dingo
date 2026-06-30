"""
Factorized sampler core.

This module implements the domain-agnostic spine of the factorized-sampler design
(see vault/Hackathon/Factorized_Sampler_Design.md): the posterior is represented as
an ordered product of conditional factors

    q(theta_1, ..., theta_n | d) = prod_i q_i(theta_i | f_i(theta_<i, d)),

where each ``Factor`` samples one block of parameters and returns its own log-prob,
and a composer drives them. The ``ChainComposer`` runs the factors autoregressively
(exact, importance-sampling-friendly); a Gibbs composer for multi-iteration GNPE is
added separately.

Key invariant -- *physical in, physical out*: factors expose only physical-space
parameters. Standardized values exist transiently inside a factor's forward pass; the
``Standardization`` helper below is the per-factor adapter between a network's internal
standardized space and physical space. Nothing we store, pass, or serialize is ever
standardized.
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
    Per-factor parameter standardization: the affine map between a network's internal
    standardized space and physical space.

    A network is trained on standardized parameters ``z = (theta - mean) / std``. This
    object holds that network's ``mean``/``std`` (read from its training metadata) and
    is used in *both* directions -- de-standardizing samples on the way out and
    standardizing parameters on the way in for ``log_prob`` -- so there is a single
    source of truth and no "inverted twice" drift.

    Standardization is a property of the *network*, not the event: different factors
    (e.g. a GNPE init network and the main network) carry different standardizations,
    and physical space is the shared interchange between them.

    Notes
    -----
    The eventual design stores ``mean``/``std`` as registered buffers inside the network
    module (for NSF, as a fixed affine transform at the flow boundary so its log-Jacobian
    folds into the density automatically). This pure-Python version is the equivalent
    seam; moving it into buffers is a follow-up that does not change the interface.
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
    Per-event shared state referenced by every factor: the data ``d`` and everything
    derived from it (the one-time prepared data representation, the likelihood, event
    metadata). Concrete implementations are domain-specific (see
    ``dingo.gw.inference.factors.GWSamplerContext``). Serialized as the transport state
    between pipe stages.
    """

    event_metadata: Optional[dict]

    def prepared_data(self) -> torch.Tensor:
        """The one-time data representation (whiten/decimate/repackage/...), computed
        once and cached. The conditioning input shared by data-conditioned factors."""
        ...

    def likelihood(self):
        """The likelihood, for likelihood-based factors (synthetic phase) and IS."""
        ...


@dataclass
class Conditioning:
    """
    What the composer hands a factor: the shared ``context`` (-> prepared data,
    likelihood, domain) plus the physical values of the earlier-block parameters this
    factor conditions on.
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
    """The vertical (depth-first) batching primitive, shared by the chain and Gibbs
    composers.

    Split ``total`` base samples into chunks of ``batch_size``, run ``run_once(n)`` -- a
    full pass through the *whole* pipeline (the entire chain, or the entire Gibbs loop) --
    per chunk, and concatenate. Working memory is bounded to one chunk carried through the
    pipeline; only the output buffer grows to ``total`` (a real run streams it to CPU per
    chunk). This is the order the old ``Sampler.run_sampler`` batches in, so parity is
    exact when ``batch_size`` matches. ``batch_size=None`` runs everything in one pass.

    ``run_once`` returns ``(samples, log_prob)``; ``log_prob`` may be ``None`` (the Gibbs
    path has no tractable density)."""
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
    A conditional distribution ``q_i(theta_i | f_i(theta_<i, d))`` over one block of
    parameters. Emits physical-space samples and a physical-space log-prob (its
    standardization Jacobian already applied), so the chain product ``sum_i log q_i`` is
    directly the physical posterior log-density.

    *Physical in, physical out*, and a **single forward pass per call** -- batching is not
    a factor concern; the composer drives it (``chunk_and_concat``).

    Contract the composer relies on (this is what makes fan-out work): a call draws
    ``num_samples`` samples **per conditioning row** and returns ``n_rows * num_samples``
    rows, flattened row-major, where ``n_rows`` is the number of rows in ``cond.given``
    (or 1 if unconditioned). This mirrors the network's own
    ``sample_and_log_prob(*context, num_samples=n) -> (B, n, dim)`` with ``B = n_rows``.

    Attributes
    ----------
    parameters : list[str]
        The block ``theta_i`` this factor produces.
    conditioning : list[str]
        Earlier-block parameter names this factor conditions on. Data dependence is
        implicit via the shared context.
    """

    parameters: list[str]
    conditioning: list[str]

    @abstractmethod
    def sample_and_log_prob(
        self, num_samples: int, cond: Conditioning
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Draw ``num_samples`` per conditioning row from ``q_i``; return ``(physical
        samples, physical-space log q_i)``. The returned dict may carry extra named
        side-channel columns (e.g. an IS kernel correction) alongside ``parameters``."""

    @abstractmethod
    def log_prob(
        self, theta_i: dict[str, torch.Tensor], cond: Conditioning
    ) -> torch.Tensor:
        """Evaluate ``log q_i(theta_i | f_i)`` at given physical ``theta_i`` (for IS /
        re-plug). One conditioning row per ``theta_i`` row (no fan-out)."""


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
    A factor wrapping a posterior model (NPE flow, FMPE, ...). Domain-agnostic: it reaches
    data only through the ``SamplerContext.prepared_data()`` protocol, so the GW-specific
    conditioning map ``f_i`` (the data preprocessing) lives entirely on the context.

    Encapsulates the network's own standardization: the public interface is physical-in /
    physical-out -- standardized values never leave the factor.

    The cheap *parameter-dependent* part of ``f_i`` (time-shift / heterodyne for GNPE) is
    a no-op here; a domain subclass overrides ``_apply_param_transforms`` to add it.
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
        """Build a factor from a model: ``parameters`` and ``context_parameters`` come from
        the model's training metadata (first-class conditioning)."""
        data_settings = _base_model_metadata(model)["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        return cls(
            model=model,
            parameters=data_settings["inference_parameters"],
            conditioning=list(context_parameters),
            context_parameters=list(context_parameters),
        )

    def sample_and_log_prob(self, num_samples, cond):
        """Draw ``num_samples`` per conditioning row. Unconditioned (plain NPE): one
        shared data context, ``num_samples`` draws. Conditioned: ``N`` context rows in
        (one embedding pass each), ``num_samples`` draws per row -> ``N * num_samples``
        rows, flattened row-major to align with the composer's ``repeat_interleave``."""
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
    """One factor in the chain plus its **fan-out**: how many samples it draws per
    incoming conditioning row. The root (first) stage ignores ``fan_out`` -- it draws the
    base count directly; a 1:1 chain link uses ``fan_out=1``; an intrinsic/extrinsic
    expansion uses ``fan_out=K`` (``K`` extrinsic draws per intrinsic sample)."""

    factor: Factor
    fan_out: int = 1


class ChainComposer:
    """
    Autoregressive composer **and vertical (depth-first) executor**: sample the stages in
    declared order (a topological order of the dependency DAG), expanding by each stage's
    fan-out, and sum their log-probs. Exact and importance-sampling-friendly. Covers plain
    NPE, single-step GNPE, prior conditioning, synthetic phase, and intrinsic/extrinsic
    splits.

    Vertical execution carries a chunk of base samples all the way down the chain, emits
    it, and discards -- bounding working memory to one chunk (the only way to draw ``M``
    intrinsic x ``K`` extrinsic without materializing ``M*K`` intermediates) and
    reproducing the old sampler's batching order (bit-exact when ``batch_size`` matches).
    Batching lives in ``chunk_and_concat``; the factors are single-pass.

    Multi-iteration (cyclic) GNPE is *not* expressible here and uses the GW
    ``GNPEGibbsComposer``. Accepts bare factors (wrapped as ``Stage(factor, fan_out=1)``) or
    explicit ``Stage``s.
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
        """The factors in order (without fan-out), for callers that only need them."""
        return [stage.factor for stage in self.stages]

    @property
    def expansion(self) -> int:
        """Product of the non-root fan-outs; total rows returned = num_samples *
        expansion (e.g. M intrinsic x K extrinsic for an extrinsic fan-out of K)."""
        return math.prod(stage.fan_out for stage in self.stages[1:])

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """``num_samples`` is the **base (root) count** -- e.g. ``M`` intrinsic samples.
        Each fan-out stage multiplies it, so the result has ``num_samples * expansion``
        rows (``M * K`` for an extrinsic ``fan_out=K``); for a plain chain (no fan-out)
        that is just ``num_samples``. ``batch_size`` chunks the base count, in the same
        (root) unit as ``num_samples``."""
        return chunk_and_concat(
            num_samples, batch_size, lambda n: self._run_chain_once(n, context)
        )

    def _run_chain_once(
        self, base: int, context: SamplerContext
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """One depth-first pass of the whole chain for ``base`` root samples."""
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
        """Full per-sample dict (parameters + ``log_prob``), for the sampler façade."""
        samples, log_prob = self.sample_and_log_prob(num_samples, context, batch_size)
        return {**samples, "log_prob": log_prob}


@runtime_checkable
class Composer(Protocol):
    """The minimal interface ``ComposedSampler`` drives: draw a per-sample dict
    (parameters, plus ``log_prob`` for density-preserving composers). ``ChainComposer``
    (here, domain-agnostic) satisfies it, as does the GW ``GNPEGibbsComposer`` (multi-iteration
    GNPE) -- which lives in ``dingo.gw.inference.factors`` because it is built entirely
    around the GNPE step (its proxies, detector times, time-shift), not in core."""

    def sample(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]: ...


class ComposedSampler:
    """
    Thin user-facing façade over a ``Composer`` (e.g. ``ChainComposer``, or the GW
    ``GNPEGibbsComposer``) and a ``SamplerContext``: runs the composer (which bounds memory via
    vertical batching), applies domain-specific post-processing, and returns samples as a
    DataFrame. The per-factor compute lives in the composer (which need only expose
    ``sample(num_samples, context, batch_size) -> dict``); this class only handles
    consolidation and post-processing -- the role the monolithic ``Sampler`` plays today.
    """

    def __init__(self, composer: Composer, context: SamplerContext):
        self.composer = composer
        self.context = context
        self.samples: Optional[pd.DataFrame] = None

    def _post_process(self, samples: dict, inverse: bool = False):
        """Hook for domain-specific post-processing (e.g. GW fixed-parameter injection
        and reference-time correction). No-op by default."""
        pass

    def run_sampler(
        self, num_samples: int, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Run the composer (vertically batched by ``batch_size``) and return samples as a
        DataFrame. ``ChainComposer.sample`` adds ``log_prob``; ``GNPEGibbsComposer``
        (multi-iteration GNPE) returns parameters + proxies with no ``log_prob``.
        ``batch_size=None`` runs in a single pass."""
        merged = self.composer.sample(num_samples, self.context, batch_size)
        merged = {k: v.cpu().numpy() for k, v in merged.items()}
        self._post_process(merged)
        self.samples = pd.DataFrame(merged)
        return self.samples
