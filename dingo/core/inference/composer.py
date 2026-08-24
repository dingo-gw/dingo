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

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

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
    if total < 1:
        raise ValueError(f"num_samples must be at least 1, got {total}.")
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


def _interleave_rows(samples, total, n):
    """Repeat each row of the carried columns `n` times (the copies of a row are
    adjacent), keeping the running log probability aligned."""
    samples = {k: v.repeat_interleave(n, 0) for k, v in samples.items()}
    if torch.is_tensor(total):
        total = total.repeat_interleave(n, 0)
    return samples, total


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
        sampled: set[str] = set()  # parameter columns (as opposed to side channels)
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
            `None` draws in one pass. For a chain rooted in a sample table, the rows
            of the result are then grouped by chunk rather than by table row.

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
