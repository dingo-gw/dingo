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
from typing import Optional, Protocol, Union, runtime_checkable

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


def _slice_given(cond: Conditioning, start: int, stop: int) -> Conditioning:
    """A view of ``cond`` whose conditioning values are restricted to rows
    ``[start:stop]``, so a factor's batch sees the conditioning aligned with the samples
    it produces. The shared context is unchanged (the prepared data is one event)."""
    return Conditioning(
        cond.context, {k: v[start:stop] for k, v in cond.given.items()}
    )


def _batch_bounds(num_samples: int, batch_size: int):
    """Yield ``(start, stop)`` chunks covering ``range(num_samples)``."""
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


class Factor(ABC):
    """
    A conditional distribution ``q_i(theta_i | f_i(theta_<i, d))`` over one block of
    parameters. Emits physical-space samples and a physical-space log-prob (its
    standardization Jacobian already applied), so the chain product ``sum_i log q_i`` is
    directly the physical posterior log-density.

    Batching is a per-factor concern: ``sample_and_log_prob`` / ``log_prob`` are
    batching wrappers that split ``num_samples`` into chunks of ``batch_size`` (slicing
    the conditioning to match), call the subclass hooks ``_sample_and_log_prob`` /
    ``_log_prob`` per chunk, and concatenate. Because the chain composer already
    materializes the full sample set between factors, each factor batches independently
    at the size that fits *its own* memory footprint -- the peak is the single most
    expensive factor at its own ``batch_size``, not the whole chain at one size.
    ``batch_size = None`` runs the whole request in one pass (identical to no batching).

    Attributes
    ----------
    parameters : list[str]
        The block ``theta_i`` this factor produces.
    conditioning : list[str]
        Earlier-block parameter names this factor conditions on. Data dependence is
        implicit via the shared context.
    batch_size : int, optional
        Max number of samples processed in a single forward pass. ``None`` -> no
        internal batching.
    """

    parameters: list[str]
    conditioning: list[str]
    batch_size: Optional[int] = None

    def sample_and_log_prob(
        self, num_samples: int, cond: Conditioning
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Draw ``theta_i ~ q_i`` (internally batched by ``batch_size``); return
        ``(physical samples, physical-space log q_i)``."""
        if not self.batch_size or self.batch_size >= num_samples:
            return self._sample_and_log_prob(num_samples, cond)
        sample_batches, lp_batches = [], []
        for start, stop in _batch_bounds(num_samples, self.batch_size):
            block, lp = self._sample_and_log_prob(
                stop - start, _slice_given(cond, start, stop)
            )
            sample_batches.append(block)
            lp_batches.append(lp)
        return _cat_dict(sample_batches), torch.cat(lp_batches)

    def log_prob(
        self, theta_i: dict[str, torch.Tensor], cond: Conditioning
    ) -> torch.Tensor:
        """Evaluate ``log q_i(theta_i | f_i)`` at given physical ``theta_i`` (for IS /
        re-plug), internally batched by ``batch_size``."""
        num_samples = next(iter(theta_i.values())).shape[0]
        if not self.batch_size or self.batch_size >= num_samples:
            return self._log_prob(theta_i, cond)
        lp_batches = []
        for start, stop in _batch_bounds(num_samples, self.batch_size):
            chunk = {k: v[start:stop] for k, v in theta_i.items()}
            lp_batches.append(self._log_prob(chunk, _slice_given(cond, start, stop)))
        return torch.cat(lp_batches)

    @abstractmethod
    def _sample_and_log_prob(
        self, num_samples: int, cond: Conditioning
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Single-pass draw of ``num_samples`` from ``q_i`` (no internal batching)."""

    @abstractmethod
    def _log_prob(
        self, theta_i: dict[str, torch.Tensor], cond: Conditioning
    ) -> torch.Tensor:
        """Single-pass evaluation of ``log q_i`` at given physical ``theta_i``."""


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
        batch_size: Optional[int] = None,
    ):
        self.model = model
        self.parameters = parameters
        self.conditioning = conditioning or []
        # Network conditioning inputs (GNPE proxies). Empty for plain NPE.
        self.context_parameters = context_parameters or []
        self.batch_size = batch_size
        std = _base_model_metadata(model)["train_settings"]["data"]["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @classmethod
    def from_model(
        cls, model: BasePosteriorModel, batch_size: Optional[int] = None
    ) -> "FlowFactor":
        """Build a factor from a model: ``parameters`` and ``context_parameters`` come from
        the model's training metadata (first-class conditioning)."""
        data_settings = _base_model_metadata(model)["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        return cls(
            model=model,
            parameters=data_settings["inference_parameters"],
            conditioning=list(context_parameters),
            context_parameters=list(context_parameters),
            batch_size=batch_size,
        )

    def _network_context(self, cond: Conditioning) -> tuple[torch.Tensor, ...]:
        """Assemble the model's positional context: the prepared data, plus standardized
        context parameters when the network conditions on proxies."""
        data = cond.context.prepared_data()
        if not self.context_parameters:
            return (data.unsqueeze(0),)
        # Standardize the (physical) conditioning values the network was trained on.
        ctx = self.standardization.standardize(cond.given, self.context_parameters)
        return data.unsqueeze(0), ctx.unsqueeze(0)

    def _sample_and_log_prob(self, num_samples, cond):
        net_context = self._network_context(cond)
        self.model.network.eval()
        with torch.no_grad():
            z, log_prob = self.model.sample_and_log_prob(
                *net_context, num_samples=num_samples
            )
        # Squeeze the batch dimension added for the (single) context.
        z = z.squeeze(0)
        log_prob = log_prob.squeeze(0)
        theta = self.standardization.destandardize(z, self.parameters)
        log_prob = log_prob + self.standardization.log_det(self.parameters)
        return theta, log_prob

    def _log_prob(self, theta_i, cond):
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


class ChainComposer:
    """
    Autoregressive composer: sample (or evaluate) the factors in declared order, which
    must be a topological order of the dependency DAG, and sum their log-probs. Exact
    and importance-sampling-friendly. Covers plain NPE, single-step GNPE, prior
    conditioning, synthetic phase, and intrinsic/extrinsic splits.

    Multi-iteration (cyclic) GNPE is *not* expressible here and uses a separate Gibbs
    composer.
    """

    def __init__(self, factors: list[Factor]):
        self.factors = factors
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

    def sample_and_log_prob(
        self, num_samples: int, context: SamplerContext
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        samples: dict[str, torch.Tensor] = {}
        total: torch.Tensor | float = 0.0
        for factor in self.factors:
            cond = Conditioning(context, {k: samples[k] for k in factor.conditioning})
            block, lp = factor.sample_and_log_prob(num_samples, cond)
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
        self, num_samples: int, context: SamplerContext
    ) -> dict[str, torch.Tensor]:
        """Full per-sample dict (parameters + ``log_prob``), for the sampler façade."""
        samples, log_prob = self.sample_and_log_prob(num_samples, context)
        return {**samples, "log_prob": log_prob}

    def set_default_batch_size(self, batch_size: Optional[int]):
        """Apply ``batch_size`` as the default to every factor that has not set its own
        (per-factor ``batch_size`` takes precedence). The façade calls this to honour a
        single ``run_sampler(batch_size=...)`` request across the whole chain."""
        for factor in self.factors:
            if factor.batch_size is None:
                factor.batch_size = batch_size


class GibbsComposer:
    """
    Multi-iteration GNPE composition. Seeds the Gibbs chain from an ``init_factor`` (e.g.
    a network predicting detector coalescence times), then iterates a GNPE step
    (``gnpe_factor.gibbs_step``) to a fixed point. The dependency is cyclic
    (theta <-> proxies), so this breaks density access: ``sample`` returns parameters
    WITHOUT a ``log_prob``. (Single-step GNPE that *does* preserve log_prob is a
    two-factor ``ChainComposer`` instead.)
    """

    def __init__(self, init_factor: Factor, gnpe_factor, num_iterations: int):
        self.init_factor = init_factor
        self.gnpe_factor = gnpe_factor
        self.num_iterations = num_iterations

    def sample(
        self, num_samples: int, context: SamplerContext
    ) -> dict[str, torch.Tensor]:
        # Seed the Gibbs chain with the init factor's parameters (e.g. detector times).
        seed, _ = self.init_factor.sample_and_log_prob(
            num_samples, Conditioning(context)
        )
        extrinsic = dict(seed)
        parameters: dict[str, torch.Tensor] = {}
        for _ in range(self.num_iterations):
            parameters, extrinsic = self.gnpe_factor.gibbs_step(
                num_samples, context, extrinsic
            )
        proxies = {
            p: extrinsic[p] for p in self.gnpe_factor.proxy_parameters if p in extrinsic
        }
        return {**parameters, **proxies}

    def set_default_batch_size(self, batch_size: Optional[int]):
        """Apply ``batch_size`` as the default to the init and GNPE factors that have not
        set their own (per-factor ``batch_size`` takes precedence)."""
        for factor in (self.init_factor, self.gnpe_factor):
            if getattr(factor, "batch_size", None) is None:
                factor.batch_size = batch_size


class ComposedSampler:
    """
    Thin user-facing façade over a composer (``ChainComposer`` or ``GibbsComposer``) and
    a ``SamplerContext``: runs the composer with optional batching, applies domain-specific
    post-processing, and returns samples as a DataFrame. The per-factor compute lives in
    the composer (which need only expose ``sample(num_samples, context) -> dict``); this
    class only handles batching, consolidation, and post-processing -- the role the
    monolithic ``Sampler`` plays today.
    """

    def __init__(
        self, composer: Union[ChainComposer, GibbsComposer], context: SamplerContext
    ):
        self.composer = composer
        self.context = context
        self.samples: Optional[pd.DataFrame] = None

    def _post_process(self, samples: dict, inverse: bool = False):
        """Hook for domain-specific post-processing (e.g. GW fixed-parameter injection
        and reference-time correction). No-op by default."""
        pass

    def _run_batch(self, num_samples: int) -> dict[str, torch.Tensor]:
        # Uniform across composers: ChainComposer.sample adds log_prob; GibbsComposer
        # (multi-iteration GNPE) returns parameters + proxies with no log_prob.
        return self.composer.sample(num_samples, self.context)

    def run_sampler(
        self, num_samples: int, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Run the chain and return samples as a DataFrame.

        Batching is per-factor (each factor splits its own ``num_samples`` by its
        ``batch_size``); ``batch_size`` here is a convenience that broadcasts a single
        default across every factor without its own, reproducing one global batch size.
        ``None`` -> no batching."""
        if batch_size is not None:
            self.composer.set_default_batch_size(batch_size)
        merged = self._run_batch(num_samples)
        merged = {k: v.cpu().numpy() for k, v in merged.items()}
        self._post_process(merged)
        self.samples = pd.DataFrame(merged)
        return self.samples
