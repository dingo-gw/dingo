"""
The chain composer and runner of the factorized sampler.

A `ChainComposer` runs a chain of steps (`dingo.core.inference.steps`) in order, building up a
table of named parameter columns together with the summed log probability, and can
re-evaluate that log probability at given samples by folding the chain in reverse.
`GibbsBlock` wraps a Gibbs loop as a single density-free step, and `ComposedSampler`
is the generic runner that returns the samples as a DataFrame. The concepts are
explained in the "Sampling chains" page of the documentation.
"""

from __future__ import annotations

import copy
from numbers import Integral
from typing import Callable, Optional, Sequence, Union

import pandas as pd
import torch

from dingo.core.inference.context import SamplerContext
from dingo.core.inference.steps import (
    Factor,
    Reparametrization,
    Step,
    TargetCorrection,
    _n_rows,
)


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
    if total < 1:
        raise ValueError(f"num_samples must be at least 1, got {total}.")
    bs = batch_size or total
    sample_parts: list[dict[str, torch.Tensor]] = []
    lp_parts: list[Optional[torch.Tensor]] = []
    for start in range(0, total, bs):
        block, lp = run_once(min(bs, total - start))
        sample_parts.append(block)
        lp_parts.append(lp)
    # Re-join the per-chunk column dicts along the row dimension.
    samples = {
        k: torch.cat([part[k] for part in sample_parts]) for k in sample_parts[0]
    }
    log_prob = None if lp_parts[0] is None else torch.cat(lp_parts)
    return samples, log_prob


def _repeat_rows(samples, total, n):
    """Repeat each row of the carried columns `n` times (the copies of a row are
    adjacent), keeping the running log probability aligned."""
    samples = {k: v.repeat_interleave(n, 0) for k, v in samples.items()}
    if torch.is_tensor(total):
        total = total.repeat_interleave(n, 0)
    return samples, total


class ChainComposer:
    """
    Runs a chain of steps in order, building up a table of samples and the summed
    proposal log probability.

    The composer holds the steps as an ordered list. At construction it checks that
    the order is consistent: every conditioning column must be produced by an
    earlier step, and no step may overwrite an existing column, except a
    `Reparametrization` replacing its own inputs. Sampling folds the steps forward,
    drawing at each step that draws and summing the log-probability contributions;
    `log_prob` folds the steps in reverse to re-evaluate the same density at given
    samples. If any step is density-free (a `GibbsBlock`), the chain has no
    tractable density and `sample` omits `log_prob`.

    This one class covers plain NPE, single-step GNPE, prior conditioning, synthetic
    phase, and, through `GibbsBlock`, multi-iteration GNPE. See the "Sampling
    chains" page of the documentation.
    """

    def __init__(self, steps: list[Step]):
        """
        Parameters
        ----------
        steps : list[Step]
            The chain, in order.

        Raises
        ------
        ValueError
            If a step conditions on a column that no earlier step produces, or would
            overwrite an existing column.
        """
        self.steps = list(steps)
        self._validate()

    def _validate(self):
        """Check that the declared order is a valid topological order of the
        conditioning DAG."""
        produced: set[str] = set()
        sampled: set[str] = set()  # parameter columns (as opposed to side channels)
        for step in self.steps:
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
            emitted = set(step.produces)
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
            if isinstance(step, TargetCorrection):
                # A correction has no inverse, so `log_prob` cannot rebuild what it
                # consumes: it may consume side-channel intermediates only.
                sampled_consumed = set(step.consumes) & sampled
                if sampled_consumed:
                    raise ValueError(
                        f"A target correction producing {step.parameters} consumes "
                        f"the sampled parameter(s) {sorted(sampled_consumed)}. A "
                        f"correction may consume only side-channel intermediates."
                    )
            produced.update(emitted)
            sampled.update(step.parameters)
            # Consumed columns leave the produced set, as in the fold, so a later step
            # may re-emit them (e.g. `RAToEventFrame` restoring a pinned `ra` that
            # `RAToTrainingFrame` consumed).
            produced.difference_update(step.consumes)
            sampled.difference_update(step.consumes)

    def sample_and_log_prob(
        self,
        num_samples: Union[int, Sequence[int]],
        context: SamplerContext,
        batch_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Draw samples from the chain.

        Parameters
        ----------
        num_samples : int or sequence of int
            How many samples to draw for each row of the table, at each step that
            draws. An int is the count for the first drawing step (typically the
            flow), after which any later drawing step draws one sample per row. This
            is the usual case: the int is the total for a chain with a single root
            row (a pin or a flow at the root), or the number per table row for a
            chain rooted in a `SampleTableFactor`. A sequence instead gives one count
            per drawing step, in chain order, so that a later step may draw several
            samples for each row it receives (for example several extrinsic draws
            per intrinsic sample). Steps that do not draw (pins, tables,
            reparametrizations, target corrections) take no count. A chain with no
            drawing step at all (a stored table run through a reparametrization,
            say) runs once, with `num_samples=1`.
        context : SamplerContext
            The per-event shared state.
        batch_size : int, optional
            Chunk the first count into batches of this size, to cap the peak memory.
            `None` draws in one pass. For a chain rooted in a sample table, the rows
            of the result are then grouped by chunk rather than by table row.

        Returns
        -------
        samples : dict[str, torch.Tensor]
            The chain's columns, with `(root rows) * prod(counts)` rows.
        log_prob : torch.Tensor or None
            The proposal log probability per row, or `None` if any step is
            density-free.

        Raises
        ------
        ValueError
            If a sequence of counts does not have one entry per drawing step, a
            count is below 1, or the chain has no drawing step and `num_samples` is
            not 1.
        """
        counts = self._sample_counts(num_samples)
        if not counts:
            # Nothing draws: the chain transforms its root rows in a single pass.
            return self._run_chain_once([], context)
        return chunk_and_concat(
            counts[0],
            batch_size,
            lambda n: self._run_chain_once([n, *counts[1:]], context),
        )

    def _sample_counts(self, num_samples: Union[int, Sequence[int]]) -> list[int]:
        """Expand `num_samples` to one count per drawing step: an int is the count
        for the first drawing step, followed by 1 for each later one. A chain with
        no drawing step takes no count (an int must then be 1)."""
        drawing = [type(step).__name__ for step in self.steps if step.draws]
        if isinstance(num_samples, Integral):
            if not drawing:
                if num_samples != 1:
                    raise ValueError(
                        f"The chain has no drawing step, so it runs once: "
                        f"num_samples must be 1, got {num_samples}."
                    )
                return []
            return [int(num_samples)] + [1] * (len(drawing) - 1)
        counts = [int(n) for n in num_samples]
        if len(counts) != len(drawing):
            raise ValueError(
                f"num_samples has {len(counts)} entries, but the chain has "
                f"{len(drawing)} drawing step(s) {drawing}. Pass an int, or one "
                f"count per drawing step."
            )
        if any(n < 1 for n in counts):
            raise ValueError(f"Every count must be at least 1, got {counts}.")
        return counts

    def _run_chain_once(
        self, counts: list[int], context: SamplerContext
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """One pass of the whole chain, with one count per drawing step.

        Every step multiplies the table by a factor `k`: its count for a step that
        draws, and 1 for one that does not. A conditioned step draws `k` samples per
        conditioning row itself; an unconditioned step is asked for the total. The
        rows carried so far are then repeated `k` times so that every row stays
        complete. Returns the samples and the summed log probability (`None` if any
        step is density-free).
        """
        samples: dict[str, torch.Tensor] = {}
        total: torch.Tensor | float | None = 0.0
        pending = iter(counts)
        for step in self.steps:
            k = next(pending) if step.draws else 1
            rows = _n_rows(samples) if samples else 1
            given = {c: samples[c] for c in step.conditioning}
            block, lp = step.sample_and_log_prob(
                k if step.conditioning else rows * k, context, given
            )
            if samples and k > 1:
                samples, total = _repeat_rows(samples, total, k)
            samples.update(block)
            for c in step.consumes:
                samples.pop(c, None)
            total = None if lp is None or total is None else total + lp
        return samples, total

    def log_prob(
        self, samples: dict[str, torch.Tensor], context: SamplerContext
    ) -> torch.Tensor:
        """Evaluate the chain's proposal log probability at given samples.

        Importance sampling does not need this: the proposal density of a chain's
        own samples is returned by `sample_and_log_prob` and stored with them. The
        reverse fold evaluates the same density at parameters the chain did not
        draw (for diagnostics, or to compare proposals), and is what makes the
        stored density a well-defined function of the final columns. The steps are
        folded in reverse order, so that the columns are restored to the
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
        num_samples: Union[int, Sequence[int]],
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

    @property
    def produces(self) -> list[str]:
        """The emitted columns (the swept parameter blocks)."""
        return self.parameters

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
            The number of walkers, one row each (the block's count in the chain).
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
        # Seed the state (the walkers are the rows), then sweep the factors: each
        # conditions on the current state and overwrites its own block, one sample
        # per walker.
        seed, _ = self.init_factor.sample_and_log_prob(num_samples, context)
        state = dict(seed)
        for _ in range(self.num_iterations):
            for factor in self.factors:
                given = {k: state[k] for k in factor.conditioning}
                block, _ = factor.sample_and_log_prob(1, context, given)
                state.update(block)
        return {p: state[p] for p in self.parameters}, None

    def describe(self) -> dict:
        """Describe the Gibbs structure, with nested descriptors for the init factor
        and the swept factors. See `Factor.describe`."""
        return {
            "step": type(self).__name__,
            "num_iterations": self.num_iterations,
            "init": self.init_factor.describe(),
            "factors": [factor.describe() for factor in self.factors],
        }


class ComposedSampler:
    """
    Runs a `ChainComposer` over a `SamplerContext` and returns the samples as a
    DataFrame.

    All domain-specific processing lives in the chain's steps and in the context, so
    the runner itself is domain-agnostic; it also assembles the provenance record of
    how the samples were made (`sampler_provenance`). The gravitational-wave
    subclass is `dingo.gw.inference.sampler.GWComposedSampler`.
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
        # The `num_samples` last requested of `run_sampler`, for the provenance
        # record: an int, or one count per drawing step.
        self.num_samples: Optional[Union[int, list[int]]] = None
        # Extra provenance merged into the record by `sampler_provenance` -- e.g. the
        # pipe adds model checkpoint paths and the density-recovery recipe.
        # Literal-only values (the settings dict round-trips through str/literal_eval).
        self.provenance_extra: dict = {}

    def run_sampler(
        self, num_samples: Union[int, Sequence[int]], batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Draw samples and store them as a DataFrame.

        Parameters
        ----------
        num_samples : int or sequence of int
            The number of samples per root row, or one count per drawing step (see
            `ChainComposer.sample_and_log_prob`).
        batch_size : int, optional
            Chunk size for drawing. `None` draws in one pass.

        Returns
        -------
        pandas.DataFrame
            One row per sample, with a `log_prob` column unless the chain is
            density-free. Also stored as `self.samples`, with the request as
            `self.num_samples`.
        """
        # Keep the request as Python literals, for the provenance record.
        self.num_samples = (
            int(num_samples)
            if isinstance(num_samples, Integral)
            else [int(n) for n in num_samples]
        )
        merged = self.composer.sample(num_samples, self.context, batch_size)
        merged = {k: v.cpu().numpy() for k, v in merged.items()}
        self.samples = pd.DataFrame(merged)
        return self.samples

    def sampler_provenance(self) -> dict:
        """Provenance of how the samples were made: the executed chain in order (one
        descriptor per step, via `Step.describe()`), the `num_samples` requested of
        `run_sampler` (an int, or one count per drawing step; absent before
        sampling), plus anything in `provenance_extra`. A domain runner stores the
        block with its exported result (the gravitational-wave subclass as
        `settings["sampler"]` of the `Result`). It is purely a record; nothing
        consumes it at load time.

        Returns
        -------
        dict
            The provenance block, of literal values only.
        """
        provenance = {"chain": [step.describe() for step in self.composer.steps]}
        if self.num_samples is not None:
            provenance["num_samples"] = self.num_samples
        provenance.update(copy.deepcopy(self.provenance_extra))
        return provenance
