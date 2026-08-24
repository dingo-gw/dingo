"""
Chain steps of the factorized sampler: factors, reparametrizations, and target
corrections.

Dingo samples the posterior with a chain of steps, run by the composer in
`dingo.core.inference.composer`. The factors among the steps write the posterior as an ordered
product of conditionals,

    q(theta_1, ..., theta_n | d) = prod_i q_i(theta_i | theta_<i, d),

each drawing one block of parameters and returning its own log probability. Other
steps reparametrize existing columns, or annotate the importance-sampling target.
Everything in this module is domain-agnostic. The gravitational-wave steps and the
per-event context live in `dingo.gw.inference`. The concepts are explained in the
"Sampling chains" page of the documentation.

Factors work in physical parameter space. A network's standardized space exists only
inside its forward pass, mediated by `Standardization`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Protocol, Union

import torch

from dingo.core.inference.context import SamplerContext
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


def _n_rows(block: dict) -> int:
    """Row count of a column block (all columns share one length)."""
    return len(next(iter(block.values())))


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
            The existing samples, one array-like column per parameter. Columns are
            cast to float32, the chain dtype (network outputs and pins are
            float32).
        log_prob : array-like, optional
            The stored log probability of each row, cast to float32. If omitted,
            the chain has no tractable density.
        """
        self.table = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in table.items()
        }
        self.table_log_prob = (
            torch.as_tensor(log_prob, dtype=torch.float32)
            if log_prob is not None
            else None
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
