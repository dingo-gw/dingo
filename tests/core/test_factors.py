"""
Unit tests for the factorized-sampler core executor (``dingo.core.factors``).

These exercise the vertical ``ChainComposer`` and ``chunk_and_concat`` with deterministic
mock factors (no networks), so they are portable and fast: fan-out expansion + conditioning
alignment, log-prob summation, topological validation, and the batching plumbing. Bit-exact
parity against the legacy samplers is a model-based check that lives with the GW models, not
here.
"""

import pytest
import torch

from dingo.core.factors import (
    ChainComposer,
    Conditioning,
    Factor,
    GibbsBlock,
    Reparametrization,
    Stage,
    TargetCorrection,
    chunk_and_concat,
)


class _ConstFactor(Factor):
    """Deterministic factor (no RNG) honoring the executor contract: draws ``n`` samples
    per conditioning row. Unconditioned, it returns ``arange(n)``; conditioned, it returns
    ``sum(given) + within_row_index/1000`` so fan-out alignment is checkable by eye. Each
    factor contributes a constant ``0.5`` to the log-prob."""

    def __init__(self, name, conditioning=()):
        self.parameters = [name]
        self.conditioning = list(conditioning)
        self._name = name

    def sample_and_log_prob(self, n, cond):
        if self.conditioning:
            base = sum(cond.given[k] for k in self.conditioning)  # (N,)
            n_rows = base.shape[0]
            # Integer-valued within-row offset: exact in float32 (no cancellation when
            # the test recovers it as b - a), and distinct per fan-out draw.
            within = torch.arange(n, dtype=base.dtype)
            vals = (base.unsqueeze(1) + within).reshape(-1)  # (N*n,), row-major
            lp = torch.full((n_rows * n,), 0.5)
        else:
            vals = torch.arange(n, dtype=torch.float32)
            lp = torch.full((n,), 0.5)
        return {self._name: vals}, lp

    def log_prob(self, theta_i, cond):
        return torch.zeros(next(iter(theta_i.values())).shape[0])


def test_root_only_draws_num_samples():
    comp = ChainComposer([_ConstFactor("a")])
    assert comp.expansion == 1
    out = comp.sample(5, context=None)
    assert torch.equal(out["a"], torch.arange(5, dtype=torch.float32))
    assert torch.allclose(out["log_prob"], torch.full((5,), 0.5))


def test_bare_factor_is_wrapped_as_stage():
    comp = ChainComposer([_ConstFactor("a")])
    assert isinstance(comp.stages[0], Stage)
    assert comp.stages[0].fan_out == 1
    assert comp.steps[0].parameters == ["a"]


def test_fan_out_expansion_alignment_and_logprob():
    m, k = 4, 3
    comp = ChainComposer(
        [
            Stage(_ConstFactor("a")),  # root -> base m
            Stage(_ConstFactor("b", conditioning=["a"]), fan_out=k),
        ]
    )
    assert comp.expansion == k
    out = comp.sample(m, context=None)  # num_samples is the base (M intrinsic)
    assert out["a"].shape == (m * k,) and out["b"].shape == (m * k,)

    # 'a' is each root value repeated k times (repeat_interleave layout).
    a = out["a"].reshape(m, k)
    assert torch.all(a == a[:, :1])
    assert torch.equal(a[:, 0], torch.arange(m, dtype=torch.float32))

    # 'b' conditions on the matching 'a': (b - a) recovers the within-row fan-out index,
    # which is distinct per draw (0..k-1) and identical across rows.
    b = out["b"].reshape(m, k)
    expected_within = torch.arange(k, dtype=torch.float32).repeat(m, 1)
    assert torch.equal(b - a, expected_within)

    # log_prob is the sum over both factors (0.5 + 0.5).
    assert torch.allclose(out["log_prob"], torch.full((m * k,), 1.0))


def test_chained_fan_out_product():
    comp = ChainComposer(
        [
            Stage(_ConstFactor("a")),
            Stage(_ConstFactor("b", conditioning=["a"]), fan_out=2),
            Stage(_ConstFactor("c", conditioning=["b"]), fan_out=5),
        ]
    )
    assert comp.expansion == 10
    out = comp.sample(3, context=None)  # base 3 -> 3 * 10 = 30 rows
    assert all(out[p].shape == (30,) for p in ("a", "b", "c"))


def test_num_samples_is_the_base_count():
    # num_samples is the base (root) count, not the total: total rows = num_samples *
    # expansion, with no divisibility constraint (here 7 is not a multiple of 4).
    m, k = 7, 4
    comp = ChainComposer(
        [
            Stage(_ConstFactor("a")),
            Stage(_ConstFactor("b", conditioning=["a"]), fan_out=k),
        ]
    )
    out = comp.sample(m, context=None)
    assert out["a"].shape == (m * k,) and out["b"].shape == (m * k,)


def test_topological_validation():
    with pytest.raises(ValueError, match="conditions on"):
        ChainComposer([_ConstFactor("b", conditioning=["a"])])  # 'a' never produced


def test_chunk_and_concat_plumbing():
    # Position-independent run_once -> batched concat is identical to a single pass.
    def run_once(n):
        return {"x": torch.full((n,), 7.0)}, torch.ones(n)

    s_full, lp_full = chunk_and_concat(10, None, run_once)
    s_chunk, lp_chunk = chunk_and_concat(10, 3, run_once)  # 3+3+3+1
    assert s_full["x"].shape == (10,)
    assert torch.equal(s_full["x"], s_chunk["x"])
    assert torch.equal(lp_full, lp_chunk)


def test_chunk_and_concat_allows_none_log_prob():
    # The Gibbs path carries no density.
    samples, lp = chunk_and_concat(6, 2, lambda n: ({"x": torch.zeros(n)}, None))
    assert lp is None
    assert samples["x"].shape == (6,)


def test_conditioning_dataclass_defaults():
    cond = Conditioning(context="ctx")
    assert cond.context == "ctx"
    assert cond.given == {}


class _NoDensityStep:
    """A density-free step (like ``GibbsBlock``): emits a block but returns ``None`` for the
    log-prob. Honors the per-row fan-out contract so it composes like any step."""

    def __init__(self, name, conditioning=()):
        self.parameters = [name]
        self.conditioning = list(conditioning)
        self._name = name

    def sample_and_log_prob(self, n, cond):
        if self.conditioning:
            base = sum(cond.given[k] for k in self.conditioning)
            within = torch.arange(n, dtype=base.dtype)
            vals = (base.unsqueeze(1) + within).reshape(-1)
        else:
            vals = torch.arange(n, dtype=torch.float32)
        return {self._name: vals}, None


def test_density_free_step_omits_log_prob():
    # A density-free root step (Gibbs) yields no proposal density.
    comp = ChainComposer([_NoDensityStep("a")])
    _, lp = comp.sample_and_log_prob(5, context=None)
    assert lp is None
    out = comp.sample(5, context=None)
    assert "log_prob" not in out
    assert torch.equal(out["a"], torch.arange(5, dtype=torch.float32))


def test_one_density_free_step_nulls_the_whole_chain():
    # A single None step makes the chain density-free even alongside a density factor.
    comp = ChainComposer(
        [_ConstFactor("a"), Stage(_NoDensityStep("b", conditioning=["a"]))]
    )
    _, lp = comp.sample_and_log_prob(4, context=None)
    assert lp is None
    assert "log_prob" not in comp.sample(4, context=None)


def test_gibbs_block_runs_as_a_density_free_step():
    # GibbsBlock seeds, sweeps its factors num_iterations times, and yields no density;
    # seed-only params (not reproduced by a factor) are dropped, matching the old composer.
    block = GibbsBlock(
        init_factor=_ConstFactor("proxy"),
        factors=[_ConstFactor("theta", conditioning=["proxy"])],
        num_iterations=3,
    )
    assert block.parameters == ["theta"] and block.conditioning == []
    comp = ChainComposer([block])
    out = comp.sample(6, context=None)
    assert "log_prob" not in out
    assert "proxy" not in out  # seed-only, dropped
    # theta | proxy with Gibbs' one-per-walker draw: theta == proxy == arange(6).
    assert torch.equal(out["theta"], torch.arange(6, dtype=torch.float32))


class _MockReparam(Reparametrization):
    """A mock bijection ``u -> v = u + shift`` with a constant nonzero ``log|det J|`` so the
    density contribution (``-log_det``) is checkable."""

    def __init__(self, shift=10.0, log_det_val=0.5):
        self.conditioning = ["u"]
        self.parameters = ["v"]
        self._shift = shift
        self._ld = log_det_val

    def forward(self, given, context):
        return {"v": given["u"] + self._shift}

    def inverse(self, params, context):
        return {"u": params["v"] - self._shift}

    def log_det(self, given, context):
        n = next(iter(given.values())).shape[0]
        return torch.full((n,), self._ld)


def test_reparam_step_consumes_input_and_contributes_neg_logdet():
    # A reparam is an in-place bijection: it replaces its input column and contributes
    # -log_det to the chain density.
    comp = ChainComposer([_ConstFactor("u"), _MockReparam(shift=10.0, log_det_val=0.5)])
    out = comp.sample(4, context=None)
    assert "u" not in out  # consumed
    assert torch.equal(out["v"], torch.arange(4, dtype=torch.float32) + 10.0)
    # chain log_prob = factor(+0.5) + reparam(-0.5) = 0.
    assert torch.allclose(out["log_prob"], torch.zeros(4))


def test_reparam_rejects_fan_out():
    rp = _MockReparam()
    with pytest.raises(ValueError, match="1:1"):
        rp.sample_and_log_prob(
            2, Conditioning(context=None, given={"u": torch.zeros(3)})
        )


def test_reparam_forward_inverse_round_trip():
    rp = _MockReparam(shift=10.0)
    u = torch.arange(5, dtype=torch.float32)
    v = rp.forward({"u": u}, context=None)["v"]
    u2 = rp.inverse({"v": v}, context=None)["u"]
    assert torch.equal(u, u2)


class _ConstFillFactor(Factor):
    """An unconditioned factor emitting a constant (like a ``FixedFactor``). As the root it
    draws the base count; as a non-root step it fills one value per current row."""

    def __init__(self, name, value=3.0):
        self.parameters = [name]
        self.conditioning = []
        self._name, self._v = name, value

    def sample_and_log_prob(self, n, cond):
        return {self._name: torch.full((n,), self._v)}, torch.zeros(n)

    def log_prob(self, theta_i, cond):
        return torch.zeros(next(iter(theta_i.values())).shape[0])


def test_unconditioned_filler_fills_the_current_batch():
    # A non-root unconditioned factor (fixed/delta filler) emits one value per current row,
    # not a single row, and contributes 0 to the density.
    comp = ChainComposer([_ConstFactor("a"), _ConstFillFactor("c", 3.0)])
    out = comp.sample(5, context=None)
    assert out["a"].shape == (5,) and out["c"].shape == (5,)
    assert torch.equal(out["c"], torch.full((5,), 3.0))
    assert torch.allclose(
        out["log_prob"], torch.full((5,), 0.5)
    )  # factor(0.5) + fill(0)


def test_unconditioned_filler_fills_after_fan_out():
    # The filler matches the expanded batch, not the base count.
    comp = ChainComposer(
        [
            _ConstFactor("a"),
            Stage(_ConstFactor("b", conditioning=["a"]), fan_out=3),
            _ConstFillFactor("c", 7.0),
        ]
    )
    out = comp.sample(4, context=None)  # 4 * 3 = 12 rows
    assert out["c"].shape == (12,)
    assert torch.equal(out["c"], torch.full((12,), 7.0))


def test_unconditioned_factor_as_root_draws_base_count():
    # As the chain root (prior-conditioning / FixedFactor proxy), an unconditioned factor
    # draws the base count; a downstream factor conditions on the pinned values.
    comp = ChainComposer(
        [_ConstFillFactor("c", 2.0), _ConstFactor("d", conditioning=["c"])]
    )
    out = comp.sample(6, context=None)
    assert out["c"].shape == (6,) and out["d"].shape == (6,)
    assert torch.equal(out["c"], torch.full((6,), 2.0))
    assert torch.equal(out["d"], torch.full((6,), 2.0))  # d | c: sum(c) + 0 == c


class _MockTargetCorrection(TargetCorrection):
    """A mock kind-3 correction: emits ``corr = 2 * x``, contributes 0 to the proposal,
    and consumes its input."""

    def __init__(self, reads="x", emits="corr"):
        self.conditioning = [reads]
        self.parameters = [emits]
        self.consumes = [reads]
        self._reads, self._emits = reads, emits

    def correction(self, given, context):
        return {self._emits: given[self._reads] * 2}


def test_target_correction_emits_side_channel_and_contributes_zero():
    comp = ChainComposer([_ConstFactor("x"), _MockTargetCorrection(reads="x")])
    out = comp.sample(5, context=None)
    assert "x" not in out  # consumed
    assert torch.equal(out["corr"], 2 * torch.arange(5, dtype=torch.float32))
    # A correction adds 0 to the proposal density: log_prob = factor(0.5) + corr(0).
    assert torch.allclose(out["log_prob"], torch.full((5,), 0.5))


def test_target_correction_rejects_fan_out():
    tc = _MockTargetCorrection()
    with pytest.raises(ValueError, match="1:1"):
        tc.sample_and_log_prob(
            2, Conditioning(context=None, given={"x": torch.zeros(3)})
        )


class _SideChannelFactor(Factor):
    """A factor emitting an extra side-channel column beyond ``parameters``, declared via
    ``produces`` so the topological check sees it (like GNPEFlowFactor's detector times).
    """

    def __init__(self, name, side):
        self.parameters = [name]
        self.conditioning = []
        self._name, self._side = name, side

    @property
    def produces(self):
        return self.parameters + [self._side]

    def sample_and_log_prob(self, n, cond):
        block = {
            self._name: torch.arange(n, dtype=torch.float32),
            self._side: torch.zeros(n),
        }
        return block, torch.zeros(n)

    def log_prob(self, theta_i, cond):
        return torch.zeros(next(iter(theta_i.values())).shape[0])


def test_side_channel_produces_satisfies_topological_check():
    # A later step may condition on a side-channel column declared via `produces`; without
    # `produces` the topological check would reject the chain.
    comp = ChainComposer(
        [_SideChannelFactor("a", side="s"), _ConstFactor("b", conditioning=["s"])]
    )
    out = comp.sample(4, context=None)
    assert out["b"].shape == (4,) and "s" in out
