"""
Core of the factorized sampler: chain steps, the chain composer, and the runner.

Dingo samples the posterior with a chain of steps. The factors among them write the
posterior as an ordered product of conditionals,

    q(theta_1, ..., theta_n | d) = prod_i q_i(theta_i | theta_<i, d),

each drawing one block of parameters and returning its own log probability. Other
steps reparametrize existing columns, or annotate the importance-sampling target. A
`ChainComposer` runs the steps in order, building up a table of named parameter
columns together with the summed log probability, and can re-evaluate that log
probability at given samples by folding the chain in reverse.

Everything in this module is domain-agnostic. The gravitational-wave steps and the
per-event context live in `dingo.gw.inference`. The concepts are explained in the
"Sampling chains" page of the documentation.

Factors work in physical parameter space. A network's standardized space exists only
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
    Affine map between a network's standardized parameter space and physical
    parameter space, `z = (theta - mean) / std`.

    Each network has its own `mean` and `std`, so each `FlowFactor` holds its own
    instance. The map is used in both directions: network outputs are
    de-standardized into physical samples, and physical parameters are standardized
    before a `log_prob` evaluation.
    """

    def __init__(self, mean: dict[str, float], std: dict[str, float]):
        """
        Parameters
        ----------
        mean : dict[str, float]
            Standardization mean, keyed by parameter name.
        std : dict[str, float]
            Standardization standard deviation, keyed by parameter name.
        """
        self.mean = dict(mean)
        self.std = dict(std)

    def standardize(
        self, values: dict[str, torch.Tensor], names: list[str]
    ) -> torch.Tensor:
        """Standardize physical parameter values.

        Parameters
        ----------
        values : dict[str, torch.Tensor]
            Physical values, keyed by parameter name.
        names : list[str]
            The parameters to include, in column order.

        Returns
        -------
        torch.Tensor
            Standardized values, with one column per entry of `names`.
        """
        cols = [(values[n] - self.mean[n]) / self.std[n] for n in names]
        return torch.stack(cols, dim=-1)

    def destandardize(
        self, z: torch.Tensor, names: list[str]
    ) -> dict[str, torch.Tensor]:
        """Map standardized values back to physical parameter values.

        Parameters
        ----------
        z : torch.Tensor
            Standardized values, with columns in `names` order.
        names : list[str]
            The parameter name of each column.

        Returns
        -------
        dict[str, torch.Tensor]
            Physical values, keyed by parameter name.
        """
        return {n: z[..., i] * self.std[n] + self.mean[n] for i, n in enumerate(names)}

    def log_det(self, names: list[str]) -> float:
        """The log-Jacobian term that converts a network log probability to physical
        parameter space, `log p(theta) = log p(z) - sum(log std)`.

        The same term is added when sampling and when evaluating `log_prob`.

        Parameters
        ----------
        names : list[str]
            The parameters included in the network's output.

        Returns
        -------
        float
            The value of `-sum(log std)` over `names`.
        """
        return -sum(math.log(self.std[n]) for n in names)


@runtime_checkable
class SamplerContext(Protocol):
    """
    Protocol for the per-event state shared by all steps of a chain.

    A context holds the event data and metadata, and everything derived from them
    that a step may need: the data in the representation the networks were trained
    on (`prepared_data`), and the likelihood. Steps never receive the data directly;
    they read it through the context. Concrete implementations are domain-specific;
    see `dingo.gw.inference.context.GWSamplerContext`.

    Attributes
    ----------
    event_metadata : dict or None
        Per-event metadata, such as the event time and analysis settings.
    device : torch.device or str
        The device the chain runs on. Steps that create new tensors, rather than
        transforming existing ones, create them on this device so that they can be
        combined with the outputs of networks running on a GPU.
    """

    event_metadata: Optional[dict]
    device: Union[torch.device, str]

    def prepared_data(self, conditioning=None) -> torch.Tensor:
        """The event data in the representation the networks condition on.

        Parameters
        ----------
        conditioning : dict[str, torch.Tensor], optional
            Chain columns available to a conditioned factor. Without it, the single
            shared representation is returned, computed once and cached. With it,
            the result has one data row per conditioning row. Only the columns the
            data preparation depends on (for example a heterodyning proxy) affect
            the result; the other columns condition the network alone.

        Returns
        -------
        torch.Tensor
        """
        ...

    def likelihood(self):
        """The likelihood of the event data, for likelihood-based factors (such as
        the synthetic phase) and for importance sampling."""
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
    """Run `run_once` in chunks and concatenate the results.

    This caps the peak memory at one chunk. The log probability may be `None`, for a
    density-free chain (one containing a `GibbsBlock`).

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


def _n_rows(block: dict) -> int:
    """Row count of a column block (all columns share one length)."""
    return len(next(iter(block.values())))


def _interleave_rows(samples, total, n):
    """Repeat each row of the carried columns `n` times (the copies of a row are
    adjacent), keeping the running log probability aligned."""
    samples = {k: v.repeat_interleave(n, 0) for k, v in samples.items()}
    if torch.is_tensor(total):
        total = total.repeat_interleave(n, 0)
    return samples, total


def _describe_default(step) -> dict:
    """Default provenance descriptor for a chain step: the class name, the parameters
    it produces, and the columns it conditions on."""
    return {
        "step": type(step).__name__,
        "parameters": list(step.parameters),
        "conditioning": list(step.conditioning),
    }


class Factor(ABC):
    """
    Base class for a factor: one conditional distribution `q_i(theta_i | theta_<i, d)`
    in the product that makes up the proposal.

    A factor draws one block of parameters and returns its own log probability, both
    in physical parameter space (any network standardization is internal). When
    conditioned, it draws `num_samples` samples for *each* row of the conditioning,
    returning `n_rows * num_samples` rows with the draws for a given conditioning row
    adjacent. This matches the convention of the posterior models,
    `sample_and_log_prob(*context, num_samples=n) -> (n_rows, n, dim)`.

    Attributes
    ----------
    parameters : list[str]
        The parameters this factor produces.
    conditioning : list[str]
        Earlier chain columns this factor conditions on. The data is not listed
        here; it enters through the context.
    draws : bool
        Whether the factor draws new samples (the default) or is a point mass or
        fixed table that is run once. The chain's `num_samples` lands on the first
        step that draws.
    consumes : tuple[str, ...]
        Columns removed from the chain after this step; none for an ordinary factor.
    """

    parameters: list[str]
    conditioning: list[str]
    draws = True
    consumes: tuple[str, ...] = ()

    @abstractmethod
    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Draw samples and evaluate their log probability.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw per conditioning row.
        context : SamplerContext
            The per-event shared state.
        given : dict[str, torch.Tensor], optional
            The conditioning columns, one row each. Omitted for an unconditioned
            factor.

        Returns
        -------
        samples : dict[str, torch.Tensor]
            The drawn values, keyed by parameter name, with `n_rows * num_samples`
            rows. The dict may also contain named columns beyond `parameters`.
        log_prob : torch.Tensor
            The log probability of each row, in physical parameter space.
        """

    @abstractmethod
    def log_prob(
        self,
        theta_i: dict[str, torch.Tensor],
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Evaluate the log probability at given parameter values.

        Parameters
        ----------
        theta_i : dict[str, torch.Tensor]
            Values of this factor's parameters, one row each.
        context : SamplerContext
            The per-event shared state.
        given : dict[str, torch.Tensor], optional
            The conditioning columns, one row per row of `theta_i`.

        Returns
        -------
        torch.Tensor
            The log probability per row, in physical parameter space.
        """

    def describe(self) -> dict:
        """Describe the step for the provenance record of a saved result.

        The default descriptor records the class name, the parameters produced, and
        the conditioning. Steps with settings worth recording override this. The
        descriptor must be literal-only: every value round-trips through `str` and
        `ast.literal_eval` in the saved settings.

        Returns
        -------
        dict
        """
        return _describe_default(self)


def _base_model_metadata(model: BasePosteriorModel) -> dict:
    """Return the analysis metadata of a model (dataset, domain, detector, and data
    settings). For an unconditional (density-recovery) model this is the metadata of
    the base model whose samples it was trained on, stored under `metadata["base"]`.
    The network's own settings (`standardization`, `inference_parameters`) are always
    read from `model.metadata` directly."""
    metadata = model.metadata
    if metadata["train_settings"]["data"].get("unconditional", False):
        return metadata["base"]
    return metadata


class FlowFactor(Factor):
    """
    Factor wrapping a posterior model (an NPE flow, FMPE, and so on).

    The factor handles the network's standardization internally, so its interface is
    in physical parameter space. Three kinds of model are supported. A
    data-conditional model draws from the shared data representation,
    `SamplerContext.prepared_data()`. A model with `context_parameters` (for example
    GNPE proxies, or a prior-conditioning pin) additionally conditions on those chain
    columns, and the data representation may depend on their values. An
    unconditional model (flagged `unconditional` in its training metadata, such as a
    density-recovery NDE) takes no input at all and does not touch the context.

    A factor may expose a trained parameter under an alias (for example `ra` as
    `ra@t_ref`), so that a later step can convert reference frames by name.
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
            Map from a trained parameter name to the name exposed in the chain (for
            example `{"ra": "ra@t_ref"}`), so that a later reparametrization can
            convert reference frames by name without retraining.
        """
        self.model = model
        # The network's trained parameter names; standardization is keyed by these.
        # The factor exposes them under their aliases (e.g. ra -> ra@t_ref).
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
        """Build a factor from a model, reading the parameter names and the context
        parameters from its training metadata. (For an unconditional NDE these are its
        own, for example the GNPE proxies it was trained on.)

        Parameters
        ----------
        model : BasePosteriorModel
            The posterior model to wrap.
        aliases : dict[str, str], optional
            Map from a trained parameter name to the name exposed in the chain (for
            example `{"ra": "ra@t_ref"}`).

        Returns
        -------
        FlowFactor
        """
        data_settings = model.metadata["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        return cls(
            model=model,
            parameters=data_settings["inference_parameters"],
            # The chain may carry a frame-corrected alias of a trained conditioning
            # name (e.g. `ra@t_ref` for a pinned event-frame `ra`).
            conditioning=[(aliases or {}).get(n, n) for n in context_parameters],
            context_parameters=list(context_parameters),
            aliases=aliases,
        )

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw samples from the model.

        A data-conditional model draws from the shared data representation; a model
        with context parameters draws `num_samples` for each row of `given`, with the
        data prepared per row; an unconditional model draws with no input. See
        `Factor.sample_and_log_prob` for the arguments and the row layout.
        """
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
            # Standardize the (physical) conditioning the network was trained on,
            # giving B = N context rows; the context returns data row-aligned to
            # them. The network draws num_samples per row -> (N, num_samples, dim).
            # TODO: the embedding runs once per row, so row-identical data with
            # row-varying conditioning (the intrinsic/extrinsic split) recomputes N
            # identical data embeddings. Fixing this needs an embed/fuse split on the
            # model (FlowWrapper) so the cached embedding can be reused across rows.
            ctx = self.standardization.standardize(
                self._network_conditioning(given), self.context_parameters
            )
            n_rows = ctx.shape[0]
            data = context.prepared_data(conditioning=given)
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
        """Evaluate the model's log probability at `theta_i`, in physical parameter
        space. See `Factor.log_prob`."""
        # theta_i uses exposed (aliased) names; map back to the network's trained names.
        theta_net = {
            net: theta_i[self.aliases.get(net, net)] for net in self._net_parameters
        }
        num_samples = _n_rows(theta_net)
        z = self.standardization.standardize(theta_net, self._net_parameters)
        net_context: tuple[torch.Tensor, ...]
        if self.unconditional:
            net_context = ()
        elif not self.context_parameters:
            data = context.prepared_data()
            data = data.expand(num_samples, *data.shape)
            net_context = (data,)
        else:
            data = context.prepared_data(conditioning=given)
            ctx = self.standardization.standardize(
                self._network_conditioning(given), self.context_parameters
            )
            net_context = (data, ctx)
        self.model.network.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(z, *net_context)
        return log_prob + self.standardization.log_det(self._net_parameters)

    def _network_conditioning(self, given):
        """The conditioning values keyed by the network's trained names. (The chain
        may carry an alias, for example `ra@t_ref` for a trained `ra`.)"""
        return {n: given[self.aliases.get(n, n)] for n in self.context_parameters}

    def describe(self) -> dict:
        """The default descriptor, plus whether the model is unconditional."""
        return {**_describe_default(self), "unconditional": self.unconditional}


class DeltaFactor(Factor):
    """
    A point mass `q_i = delta(theta_i - c)` that pins parameters to fixed values.

    A delta factor is used in two ways: as the root of a chain, supplying pinned
    values (a known proxy, or a prior-conditioning pin) that later factors condition
    on; and as a filler, supplying delta-prior parameters that the network does not
    infer. It contributes zero to the proposal log probability: the chain's log
    probability covers only the parameters that are sampled, and the pinned block is
    conditioned on rather than integrated over. The importance-sampling target uses
    the same convention, so the factor cancels in the weights.

    Every draw returns the same values, so the factor does not draw
    (`draws = False`). As a root it is run once, and the chain's `num_samples` is
    drawn by the first step that does draw.
    """

    draws = False

    def __init__(self, values: dict[str, float]):
        """
        Parameters
        ----------
        values : dict[str, float]
            The pinned value of each parameter.
        """
        self.values = values
        self.parameters = list(values)
        self.conditioning = []

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Return `num_samples` copies of the pinned values, with zero log probability.
        See `Factor.sample_and_log_prob`."""
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
        """Return zero for every row.

        The chain only evaluates `log_prob` at its own samples, so the point mass is
        always evaluated on its support; values off the support are not represented.
        See `Factor.log_prob`.
        """
        # One zero per row, on the same device/dtype as the evaluated block.
        reference_column = next(iter(theta_i.values()))
        return torch.zeros_like(reference_column)

    def describe(self) -> dict:
        """The default descriptor, plus the pinned values."""
        return {
            **_describe_default(self),
            "values": {k: float(v) for k, v in self.values.items()},
        }


class SampleTableFactor(Factor):
    """
    A chain root that emits a fixed table of existing samples, together with their
    stored log probability.

    Use this to continue a chain from samples drawn earlier. For example, the
    synthetic phase is added to previously drawn samples by a chain rooted in their
    table, and the chain's summed log probability is then the joint proposal density
    `log q(theta) + log q(phase | theta)`. Without a stored log probability the chain
    is density-free.

    Unlike a `DeltaFactor`, a sample table is not a distribution: it carries the
    density of the chain that produced its rows and cannot be evaluated at other
    points, so its `log_prob` raises. Like a delta factor it does not draw
    (`draws = False`). The table is emitted once, and the chain's `num_samples` is
    drawn per table row by the first step that does draw (one phase per sample, or
    `num_samples` posterior draws per grid point in the chirp-mass scan).
    """

    draws = False

    def __init__(self, table: dict, log_prob=None):
        """
        Parameters
        ----------
        table : dict
            The existing samples, one array-like column per parameter.
        log_prob : array-like, optional
            The stored log probability of each row. If omitted, the chain has no
            tractable density.
        """
        self.table = {k: torch.as_tensor(v) for k, v in table.items()}
        self.table_log_prob = (
            torch.as_tensor(log_prob) if log_prob is not None else None
        )
        self.parameters = list(self.table)
        self.conditioning: list[str] = []

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Emit the table and its stored log probability.

        Parameters
        ----------
        num_samples : int
            Must be 1: the table is emitted once.
        context : SamplerContext
            The per-event shared state. The table is moved to its device.
        given : dict, optional
            Ignored; a table is unconditioned.

        Returns
        -------
        samples : dict[str, torch.Tensor]
            The table columns.
        log_prob : torch.Tensor or None
            The stored log probability per row, or `None` if none was given.

        Raises
        ------
        ValueError
            If `num_samples` is not 1.
        """
        if num_samples != 1:
            raise ValueError(
                f"A sample table is emitted once (num_samples=1), got {num_samples}. "
                f"The chain's num_samples is drawn per table row by the first step "
                f"that draws."
            )
        # The table's fresh tensors join the chain on its device (the same policy
        # as DeltaFactor), so a table-rooted chain can condition a CUDA network.
        device = getattr(context, "device", None)
        table = {k: v.to(device) for k, v in self.table.items()}
        log_prob = (
            self.table_log_prob.to(device) if self.table_log_prob is not None else None
        )
        return table, log_prob

    def log_prob(self, theta_i, context, given=None):
        """Raise `NotImplementedError`: a table is not a density. Evaluate the log
        probability through the chain that produced the samples instead."""
        raise NotImplementedError(
            "A sample table is not a density; its rows carry their stored log-prob. "
            "Evaluate log_prob through the chain that produced the samples instead."
        )


class Reparametrization(ABC):
    """
    Base class for a reparametrization: a deterministic, invertible change of
    variables applied to existing chain columns.

    A reparametrization does not sample. Its `forward` map takes the conditioning
    columns to the `parameters` it produces, replacing the inputs it `consumes`, and
    contributes `-log|det J|` to the proposal log probability (zero for a
    measure-preserving map, the default). It is one-to-one, with one output row per
    input row, so it carries no sample multiplicity. The `inverse` map rebuilds the
    consumed inputs, which is what lets `ChainComposer.log_prob` re-evaluate a chain
    at given samples. Typical uses relate a network's coordinates to physical ones,
    such as rotating the right ascension from the training reference frame to the
    event frame.

    Subclasses implement `forward` and `inverse`, and `log_det` when the map is not
    measure-preserving.

    Attributes
    ----------
    parameters : list[str]
        The columns produced.
    conditioning : list[str]
        The columns read.
    """

    parameters: list[str]
    conditioning: list[str]
    draws = False

    @abstractmethod
    def forward(
        self, given: dict[str, torch.Tensor], context: "SamplerContext"
    ) -> dict[str, torch.Tensor]:
        """Apply the change of variables.

        Parameters
        ----------
        given : dict[str, torch.Tensor]
            The conditioning columns, one row each.
        context : SamplerContext
            The per-event shared state.

        Returns
        -------
        dict[str, torch.Tensor]
            The `parameters` columns.
        """

    @abstractmethod
    def inverse(
        self,
        params: dict[str, torch.Tensor],
        context: "SamplerContext",
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Rebuild the consumed inputs from the produced parameters.

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            The `parameters` columns, one row each.
        context : SamplerContext
            The per-event shared state.
        given : dict[str, torch.Tensor], optional
            The conditioning columns that were not consumed and are still in the
            chain, for example a proxy the map shifts by. Maps that depend only on
            their own outputs may ignore it.

        Returns
        -------
        dict[str, torch.Tensor]
            The consumed columns.
        """

    def log_det(
        self, given: dict[str, torch.Tensor], context: "SamplerContext"
    ) -> torch.Tensor:
        """The log-Jacobian `log|det J|` of `forward`, per row.

        The default is zero, for a measure-preserving map.

        Parameters
        ----------
        given : dict[str, torch.Tensor]
            The conditioning columns, one row each.
        context : SamplerContext
            The per-event shared state.

        Returns
        -------
        torch.Tensor
            One value per row, on the device of the transformed rows.
        """
        reference_column = next(iter(given.values()))
        return torch.zeros_like(reference_column)

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Apply `forward` and contribute `-log|det J|`.

        `num_samples` must be 1, since a reparametrization is one-to-one. See
        `Factor.sample_and_log_prob` for the arguments.

        Raises
        ------
        ValueError
            If `num_samples` is not 1.
        """
        if num_samples != 1:
            raise ValueError("A reparametrization is 1:1; use fan_out=1.")
        out = self.forward(given, context)
        return out, -self.log_det(given, context)

    @property
    def consumes(self) -> list[str]:
        """The conditioning columns replaced by the outputs, and dropped from the
        chain after the step. By default, every conditioning column that is not also
        produced. `ChainComposer.log_prob` rebuilds them via `inverse`."""
        return [c for c in self.conditioning if c not in self.parameters]

    def describe(self) -> dict:
        """Describe the step for the provenance record of a saved result. See
        `Factor.describe`."""
        return _describe_default(self)


class ProxyOffsetReparam(Reparametrization):
    """
    Reconstruct a physical parameter from a network's offset output and its proxy,
    `X = delta_X + X_proxy`.

    A proxy-conditioned network (for example the chirp-mass prior conditioning of
    DINGO-BNS) infers the offset `delta_X = X - X_proxy` rather than `X` itself.
    This step rebuilds `X`. It consumes the offset column and keeps the proxy in the
    chain, where it is recorded with the samples (like the GNPE time proxies). At a
    fixed proxy the map is a pure shift, so `log_det` is zero; `inverse` recovers the
    offset from the proxy, which the reverse fold supplies.
    """

    def __init__(self, parameter_name: str):
        """
        Parameters
        ----------
        parameter_name : str
            The physical parameter name `X`. The step reads `delta_X` and `X_proxy`
            and produces `X`.
        """
        self.parameter_name = parameter_name
        self.delta_name = f"delta_{parameter_name}"
        self.proxy_name = f"{parameter_name}_proxy"
        self.parameters = [parameter_name]
        self.conditioning = [self.delta_name, self.proxy_name]

    @property
    def consumes(self) -> list[str]:
        # The offset is replaced by the physical parameter; the proxy stays in
        # the chain (recorded with the samples).
        return [self.delta_name]

    def forward(self, given, context):
        return {self.parameter_name: given[self.delta_name] + given[self.proxy_name]}

    def inverse(self, params, context, given=None):
        if given is None or self.proxy_name not in given:
            raise ValueError(
                f"Inverting {self.parameter_name} = {self.delta_name} + "
                f"{self.proxy_name} requires the proxy in `given`."
            )
        return {self.delta_name: params[self.parameter_name] - given[self.proxy_name]}


class TargetCorrection(ABC):
    """
    Base class for a target correction: a step that annotates the importance-sampling
    target and contributes nothing to the proposal.

    Some targets are not simply prior times likelihood. A target correction emits a
    side-channel column (`delta_log_prob_target` in Dingo's use), which importance
    sampling adds to the target log density; the step contributes zero to the
    proposal. It is one-to-one, with one output row per input row. It reads earlier
    columns and may consume intermediates it no longer needs.

    A target correction has no inverse. Unlike a `Reparametrization` it may therefore
    consume only side-channel intermediates (for example detector times computed by
    an earlier step), never a sampled parameter, since `ChainComposer.log_prob` could
    not rebuild it.

    Attributes
    ----------
    parameters : list[str]
        The column(s) emitted.
    conditioning : list[str]
        The columns read.
    consumes : list[str]
        The intermediate columns dropped after the step.
    """

    parameters: list[str]
    conditioning: list[str]
    draws = False
    consumes: list[str]

    @abstractmethod
    def correction(
        self, given: dict[str, torch.Tensor], context: "SamplerContext"
    ) -> dict[str, torch.Tensor]:
        """Compute the correction column(s).

        Parameters
        ----------
        given : dict[str, torch.Tensor]
            The conditioning columns, one row each.
        context : SamplerContext
            The per-event shared state.

        Returns
        -------
        dict[str, torch.Tensor]
            The emitted column(s), one value per row.
        """

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Emit the correction, with zero proposal log probability.

        `num_samples` must be 1, since a target correction is one-to-one. See
        `Factor.sample_and_log_prob` for the arguments.

        Raises
        ------
        ValueError
            If `num_samples` is not 1.
        """
        if num_samples != 1:
            raise ValueError("A target correction is 1:1; use fan_out=1.")
        out = self.correction(given, context)
        # 0 proposal contribution per row, on the device of the emitted column.
        reference_column = next(iter(out.values()))
        return out, torch.zeros_like(reference_column)

    def describe(self) -> dict:
        """Describe the step for the provenance record of a saved result. See
        `Factor.describe`."""
        return _describe_default(self)


class Step(Protocol):
    """
    Protocol for one entry of a chain: anything a `ChainComposer` can fold over.

    A step names the columns it produces (`parameters`) and the earlier columns it
    reads (`conditioning`), and implements `sample_and_log_prob`. A factor returns a
    log-probability tensor; a density-free block (`GibbsBlock`) returns `None`. The
    data is not part of the interface; steps read it through the context.

    Attributes
    ----------
    parameters : list[str]
        The columns produced.
    conditioning : list[str]
        The earlier columns read.
    draws : bool
        Whether the step draws new samples, or is run once (a point mass, a sample
        table, a one-to-one transform).
    consumes : list[str] or tuple[str, ...]
        Columns removed from the chain after the step.
    """

    parameters: list[str]
    conditioning: list[str]
    draws: bool
    consumes: Union[list[str], tuple[str, ...]]

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        given: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Produce the step's columns and its proposal log-probability contribution
        (`None` for a density-free step). See `Factor.sample_and_log_prob`."""
        ...


@dataclass
class Stage:
    """A chain step together with its fan-out.

    Every stage multiplies the table by its `fan_out`: for each row so far, `fan_out`
    samples are drawn (a conditioned step draws them per conditioning row itself; an
    unconditioned step is asked for the total), and the earlier columns are repeated
    to match. The first stage that draws uses `num_samples` times its `fan_out`.
    `fan_out` must be 1 on the root stage, where it would only multiply
    `num_samples`, and on non-drawing steps.

    Attributes
    ----------
    step : Step
        The chain step.
    fan_out : int
        Samples drawn per row of the table so far. Default 1.
    """

    step: Step
    fan_out: int = 1


class ChainComposer:
    """
    Runs a chain of steps in order, building up a table of samples and the summed
    proposal log probability.

    The composer holds an ordered list of `Stage` entries (bare steps are accepted,
    and wrapped with `fan_out=1`). At construction it checks that the order is
    consistent: every conditioning column must be produced by an earlier step, and
    no step may overwrite an existing column, except a `Reparametrization` replacing
    its own inputs. Sampling folds the steps forward, expanding the table by each
    stage's fan-out and summing the log-probability contributions; `log_prob` folds
    the steps in reverse to re-evaluate the same density at given samples. If any
    step is density-free (a `GibbsBlock`), the chain has no tractable density and
    `sample` omits `log_prob`.

    This one class covers plain NPE, single-step GNPE, prior conditioning, synthetic
    phase, and, through `GibbsBlock`, multi-iteration GNPE. See the "Sampling
    chains" page of the documentation.
    """

    def __init__(self, stages: list[Union["Stage", Step]]):
        """
        Parameters
        ----------
        stages : list[Stage or Step]
            The chain, in order. Bare steps are wrapped as `Stage(step, fan_out=1)`.

        Raises
        ------
        ValueError
            If a step conditions on a column that no earlier step produces, would
            overwrite an existing column, or has a `fan_out` other than 1 where it
            has no effect (the root stage, or a non-drawing step).
        """
        self.stages = [s if isinstance(s, Stage) else Stage(s) for s in stages]
        self._validate()

    def _validate(self):
        """Check that the declared order is a valid topological order of the
        conditioning DAG."""
        produced: set[str] = set()
        for i, stage in enumerate(self.stages):
            step = stage.step
            if stage.fan_out != 1 and (i == 0 or not step.draws):
                raise ValueError(
                    f"Stage {i} ({type(step).__name__}) has fan_out={stage.fan_out}, "
                    f"but fan_out must be 1 on the root stage (it would only multiply "
                    f"num_samples) and on non-drawing steps (point masses, sample "
                    f"tables, reparametrizations, target corrections)."
                )
            missing = [c for c in step.conditioning if c not in produced]
            if missing:
                raise ValueError(
                    f"A step producing {step.parameters} conditions on {missing}, "
                    f"which no earlier step produces. Check chain order."
                )
            # A step's emitted columns default to its `parameters`; a step may also
            # emit side-channel columns (`produces`). Only a Reparametrization may
            # overwrite existing columns (its own inputs): it is invertible, so
            # `log_prob` can restore them.
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
            # Consumed columns leave the produced set, as in the fold, so a later step
            # may re-emit them (e.g. `RAToEventFrame` restoring a pinned `ra` that
            # `RAToTrainingFrame` consumed).
            produced.difference_update(step.consumes)

    @property
    def steps(self) -> list[Step]:
        """The stage steps, in order."""
        return [stage.step for stage in self.stages]

    @property
    def expansion(self) -> int:
        """Product of the fan-outs of the stages after the root. `sample` returns
        `num_samples * expansion` rows per root row."""
        return math.prod(stage.fan_out for stage in self.stages[1:])

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Draw samples from the chain.

        Parameters
        ----------
        num_samples : int
            The number of samples drawn per root row. A chain rooted in a pin or a
            flow has a single root row, so this is the total; a chain rooted in a
            `SampleTableFactor` draws `num_samples` per table row.
        context : SamplerContext
            The per-event shared state.
        batch_size : int, optional
            Chunk `num_samples` into batches of this size, to cap the peak memory.
            `None` draws in one pass.

        Returns
        -------
        samples : dict[str, torch.Tensor]
            The chain's columns, with `(root rows) * num_samples * expansion` rows.
        log_prob : torch.Tensor or None
            The proposal log probability per row, or `None` if any step is
            density-free.
        """
        return chunk_and_concat(
            num_samples, batch_size, lambda n: self._run_chain_once(n, context)
        )

    def _run_chain_once(
        self, num_samples: int, context: SamplerContext
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """One pass of the whole chain, for `num_samples` samples per root row.

        Every stage multiplies the table by a factor `k`: 1 for a non-drawing step,
        `num_samples` times the stage's fan-out for the first step that draws, and
        the fan-out for later stages. A conditioned step draws `k` samples per
        conditioning row itself; an unconditioned step is asked for the total. The
        rows carried so far are then repeated `k` times so that every row stays
        complete. Returns the samples and the summed log probability (`None` if any
        step is density-free).
        """
        samples: dict[str, torch.Tensor] = {}
        total: torch.Tensor | float | None = 0.0
        pending: Optional[int] = num_samples  # lands on the first step that draws
        for stage in self.stages:
            step = stage.step
            if not step.draws:
                k = 1
            elif pending is not None:
                k, pending = pending * stage.fan_out, None
            else:
                k = stage.fan_out
            rows = _n_rows(samples) if samples else 1
            given = {c: samples[c] for c in step.conditioning}
            block, lp = step.sample_and_log_prob(
                k if step.conditioning else rows * k, context, given
            )
            if samples and k > 1:
                samples, total = _interleave_rows(samples, total, k)
            samples.update(block)
            for c in step.consumes:
                samples.pop(c, None)
            total = None if lp is None or total is None else total + lp
        if pending is not None and pending > 1:
            # No step draws (a chain of pins only): repeat the rows instead.
            samples, total = _interleave_rows(samples, total, pending)
        return samples, total

    def log_prob(
        self, samples: dict[str, torch.Tensor], context: SamplerContext
    ) -> torch.Tensor:
        """Evaluate the chain's proposal log probability at given samples.

        This is used to re-evaluate saved samples, and for importance sampling. The
        steps are folded in reverse order, so that the columns are restored to the
        state each step saw during sampling: a `Reparametrization` rebuilds the
        inputs it consumed via `inverse` (for example `ra@t_ref` from the event-frame
        `ra`) and contributes `-log|det J|`; a `Factor` contributes its `log_prob` at
        the restored conditioning; a `TargetCorrection` contributes nothing.

        Parameters
        ----------
        samples : dict[str, torch.Tensor]
            The chain's emitted columns, one value per row. Consumed intermediates
            are rebuilt by the reparametrization inverses and need not be present.
        context : SamplerContext
            The per-event shared state.

        Returns
        -------
        torch.Tensor
            The proposal log probability per row.

        Raises
        ------
        ValueError
            If the chain is density-free (contains a `GibbsBlock`).
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
        """Draw samples and return them as a single dict.

        The arguments are those of `sample_and_log_prob`.

        Returns
        -------
        dict[str, torch.Tensor]
            The chain's columns, plus a `log_prob` entry unless the chain is
            density-free.
        """
        samples, log_prob = self.sample_and_log_prob(num_samples, context, batch_size)
        if log_prob is None:
            return dict(samples)
        return {**samples, "log_prob": log_prob}


class GibbsBlock:
    """
    A step that runs blocked Gibbs sampling internally and yields no log probability.

    The block seeds its state from an init factor, then sweeps its list of factors in
    order for `num_iterations` iterations; each factor conditions on the current
    state and overwrites its own block. Because of the cyclic dependency the result
    has no tractable density, so the step returns `None` for the log probability, and
    a chain containing it is *density-free*: its samples carry no `log_prob`, and the
    density must be recovered afterwards (by fitting an unconditional model to the
    samples and taking a chain step with it). Dingo uses this for multi-iteration
    GNPE, with the factors in `dingo.gw.inference.steps`, but the loop is generic.

    Batching is handled by the enclosing `ChainComposer`, which runs the whole loop
    for each chunk of walkers.
    """

    draws = True
    consumes: tuple[str, ...] = ()

    def __init__(self, init_factor: Factor, factors: list[Factor], num_iterations: int):
        """
        Parameters
        ----------
        init_factor : Factor
            Seeds the state (for example an init network's detector times).
        factors : list[Factor]
            The factors swept in order in each iteration. Each conditions on the
            current state and overwrites its own block.
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
        """Run the Gibbs loop.

        Parameters
        ----------
        num_samples : int
            The number of walkers. Gibbs does not fan out; each walker is one row.
        context : SamplerContext
            The per-event shared state.
        given : dict, optional
            Ignored; the block is unconditioned.

        Returns
        -------
        samples : dict[str, torch.Tensor]
            The swept parameter blocks, one row per walker.
        log_prob : None
        """
        return self._run_once(num_samples, context), None

    def describe(self) -> dict:
        """Describe the Gibbs structure, with nested descriptors for the init factor
        and the swept factors. See `Factor.describe`."""
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
    Runs a `ChainComposer` over a `SamplerContext` and returns the samples as a
    DataFrame.

    All domain-specific processing lives in the chain's steps and in the context, so
    the runner itself is domain-agnostic. The gravitational-wave subclass is
    `dingo.gw.inference.sampler.GWComposedSampler`.
    """

    def __init__(self, composer: ChainComposer, context: SamplerContext):
        """
        Parameters
        ----------
        composer : ChainComposer
            The chain to run.
        context : SamplerContext
            The per-event shared state.
        """
        self.composer = composer
        self.context = context
        self.samples: Optional[pd.DataFrame] = None

    def run_sampler(
        self, num_samples: int, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Draw samples and store them as a DataFrame.

        Parameters
        ----------
        num_samples : int
            The number of samples per root row (see
            `ChainComposer.sample_and_log_prob`).
        batch_size : int, optional
            Chunk size for drawing. `None` draws in one pass.

        Returns
        -------
        pandas.DataFrame
            One row per sample, with a `log_prob` column unless the chain is
            density-free. Also stored as `self.samples`.
        """
        merged = self.composer.sample(num_samples, self.context, batch_size)
        merged = {k: v.cpu().numpy() for k, v in merged.items()}
        self.samples = pd.DataFrame(merged)
        return self.samples
