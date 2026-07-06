"""
Unit tests for the factorized-sampler core executor (``dingo.core.factors``).

These exercise the vertical ``ChainComposer`` and ``chunk_and_concat`` with deterministic
mock factors (no networks), so they are portable and fast: fan-out expansion + conditioning
alignment, log-prob summation, topological validation, and the batching plumbing. Bit-exact
parity against the legacy samplers is a model-based check that lives with the GW models, not
here.
"""

import math

import pytest
import torch

from dingo.core.factors import (
    ChainComposer,
    DeltaFactor,
    Factor,
    FlowFactor,
    GibbsBlock,
    Reparametrization,
    SampleTableFactor,
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

    def sample_and_log_prob(self, n, context, given=None):
        if self.conditioning:
            base = sum(given[k] for k in self.conditioning)  # (N,)
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

    def log_prob(self, theta_i, context, given=None):
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


class _NoDensityStep:
    """A density-free step (like ``GibbsBlock``): emits a block but returns ``None`` for the
    log-prob. Honors the per-row fan-out contract so it composes like any step."""

    def __init__(self, name, conditioning=()):
        self.parameters = [name]
        self.conditioning = list(conditioning)
        self._name = name

    def sample_and_log_prob(self, n, context, given=None):
        if self.conditioning:
            base = sum(given[k] for k in self.conditioning)
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


def test_sample_table_root_feeds_chain_and_sums_log_prob():
    # A chain continuing from existing samples: the table root emits the rows with
    # their stored log-prob, a 1:1 factor adds a block, and the ordinary fold yields
    # the joint proposal density (stored + delta) -- the importance-sampling shape.
    stored = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    table = SampleTableFactor(
        {"a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}, log_prob=stored
    )

    class _PlusOne(Factor):
        parameters = ["b"]
        conditioning = ["a"]

        def sample_and_log_prob(self, n, context, given=None):
            assert n == 1  # 1:1
            return {"b": given["a"] + 1.0}, torch.full_like(given["a"], 0.25)

        def log_prob(self, theta_i, context, given=None):
            return torch.full_like(given["a"], 0.25)

    out, lp = ChainComposer([table, _PlusOne()]).sample_and_log_prob(3, None)
    assert torch.equal(out["a"], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert torch.equal(out["b"], torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    assert torch.equal(lp, stored + 0.25)


def test_sample_table_requires_full_length():
    table = SampleTableFactor({"a": torch.arange(3.0)})
    with pytest.raises(ValueError, match="table length"):
        table.sample_and_log_prob(2, None)


def test_sample_table_without_log_prob_is_density_free():
    table = SampleTableFactor({"a": torch.arange(3.0)})
    _, lp = ChainComposer([table]).sample_and_log_prob(3, None)
    assert lp is None


def test_sample_table_log_prob_raises():
    table = SampleTableFactor({"a": torch.arange(3.0)})
    with pytest.raises(NotImplementedError, match="stored log-prob"):
        table.log_prob({"a": torch.zeros(3)}, None)


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
        rp.sample_and_log_prob(2, None, {"u": torch.zeros(3)})


def test_reparam_forward_inverse_round_trip():
    rp = _MockReparam(shift=10.0)
    u = torch.arange(5, dtype=torch.float32)
    v = rp.forward({"u": u}, context=None)["v"]
    u2 = rp.inverse({"v": v}, context=None)["u"]
    assert torch.equal(u, u2)


class _ConstFillFactor(Factor):
    """An unconditioned factor emitting a constant (like a ``DeltaFactor``). As the root it
    draws the base count; as a non-root step it fills one value per current row."""

    def __init__(self, name, value=3.0):
        self.parameters = [name]
        self.conditioning = []
        self._name, self._v = name, value

    def sample_and_log_prob(self, n, context, given=None):
        return {self._name: torch.full((n,), self._v)}, torch.zeros(n)

    def log_prob(self, theta_i, context, given=None):
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
    # As the chain root (prior-conditioning / DeltaFactor proxy), an unconditioned factor
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
    and (by default) consumes its input."""

    def __init__(self, reads="x", emits="corr", consume=True):
        self.conditioning = [reads]
        self.parameters = [emits]
        self.consumes = [reads] if consume else []
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
        tc.sample_and_log_prob(2, None, {"x": torch.zeros(3)})


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

    def sample_and_log_prob(self, n, context, given=None):
        block = {
            self._name: torch.arange(n, dtype=torch.float32),
            self._side: torch.zeros(n),
        }
        return block, torch.zeros(n)

    def log_prob(self, theta_i, context, given=None):
        return torch.zeros(next(iter(theta_i.values())).shape[0])


def test_side_channel_produces_satisfies_topological_check():
    # A later step may condition on a side-channel column declared via `produces`; without
    # `produces` the topological check would reject the chain.
    comp = ChainComposer(
        [_SideChannelFactor("a", side="s"), _ConstFactor("b", conditioning=["s"])]
    )
    out = comp.sample(4, context=None)
    assert out["b"].shape == (4,) and "s" in out


class _GaussFactor(Factor):
    """A factor with a real analytic density -- standard normal around the sum of its
    conditioning (0 if unconditioned) -- so log_prob re-plug is checkable against the
    log-prob stored at sample time."""

    def __init__(self, name, conditioning=()):
        self.parameters = [name]
        self.conditioning = list(conditioning)
        self._name = name

    def _mean(self, given, n_rows):
        if self.conditioning:
            return sum(given[k] for k in self.conditioning)
        return torch.zeros(n_rows)

    def sample_and_log_prob(self, n, context, given=None):
        n_rows = 1 if not self.conditioning else next(iter(given.values())).shape[0]
        mean = self._mean(given, n_rows).repeat_interleave(n)
        vals = mean + torch.randn(n_rows * n)
        return {self._name: vals}, self.log_prob({self._name: vals}, context, given)

    def log_prob(self, theta_i, context, given=None):
        v = theta_i[self._name]
        mean = self._mean(given, v.shape[0])
        return -0.5 * (v - mean) ** 2 - 0.5 * math.log(2 * math.pi)


def test_log_prob_replug_kind_aware():
    # Re-plug: the kind-aware log_prob rebuilds consumed columns via the reparam
    # inverses, sums factor densities and Jacobians, and skips target corrections.
    torch.manual_seed(0)
    comp = ChainComposer(
        [
            _GaussFactor("u"),
            _MockReparam(shift=10.0, log_det_val=0.5),
            _MockTargetCorrection(reads="v", emits="corr", consume=False),
            _GaussFactor("w", conditioning=["v"]),
        ]
    )
    out = comp.sample(64, context=None)
    assert "u" not in out  # consumed by the reparam
    samples = {k: v for k, v in out.items() if k != "log_prob"}
    lp = comp.log_prob(samples, context=None)
    # u is rebuilt as v - shift; the float round trip is not bit-exact, so compare
    # with a tolerance.
    assert torch.allclose(lp, out["log_prob"], atol=1e-5)
    # And against a hand-built sum: gauss(u) + gauss(w | v) - log_det.
    manual = (
        comp.steps[0].log_prob({"u": samples["v"] - 10.0}, None)
        + comp.steps[3].log_prob({"w": samples["w"]}, None, {"v": samples["v"]})
        - 0.5
    )
    assert torch.allclose(lp, manual, atol=1e-6)


def test_log_prob_raises_for_density_free_chain():
    block = GibbsBlock(
        init_factor=_ConstFactor("proxy"),
        factors=[_ConstFactor("theta", conditioning=["proxy"])],
        num_iterations=2,
    )
    comp = ChainComposer([block])
    out = comp.sample(4, context=None)
    with pytest.raises(ValueError, match="density-free"):
        comp.log_prob(dict(out), context=None)


class _InPlaceReparam(Reparametrization):
    """A bijection that replaces its input under the same name (``x -> x + shift``),
    so evaluation must restore the column state per step position."""

    def __init__(self, shift=5.0):
        self.conditioning = ["x"]
        self.parameters = ["x"]
        self._shift = shift

    def forward(self, given, context):
        return {"x": given["x"] + self._shift}

    def inverse(self, params, context):
        return {"x": params["x"] - self._shift}


def test_log_prob_replug_in_place_reparam():
    # An in-place reparam overwrites its own column; the reverse fold restores the
    # pre-transform value before the factor is scored, and a downstream consumer
    # (the correction) is evaluated against the post-transform value it saw.
    torch.manual_seed(0)
    comp = ChainComposer(
        [
            _GaussFactor("x"),
            _InPlaceReparam(shift=5.0),
            _MockTargetCorrection(reads="x", emits="corr", consume=False),
        ]
    )
    out = comp.sample(64, context=None)
    assert torch.equal(out["corr"], 2 * out["x"])  # correction saw the shifted x
    samples = {k: v for k, v in out.items() if k != "log_prob"}
    lp = comp.log_prob(samples, context=None)
    assert torch.allclose(lp, out["log_prob"], atol=1e-5)


def test_validation_rejects_column_overwrite():
    # Only a Reparametrization may replace an existing column (it is invertible);
    # a factor re-producing a name would destroy state log_prob cannot rebuild.
    with pytest.raises(ValueError, match="overwrite"):
        ChainComposer([_ConstFactor("a"), _ConstFactor("a")])
    # The in-place reparam is the allowed exception.
    ChainComposer([_ConstFactor("x"), _InPlaceReparam()])


class _DeviceContext:
    """Bare context stub carrying only a device (these steps ignore the rest)."""

    def __init__(self, device):
        self.device = device
        self.event_metadata = None


def test_delta_factor_creates_on_context_device():
    # A DeltaFactor creates fresh tensors, so they land on the chain's device;
    # the 'meta' device stands in for a GPU without needing one.
    delta = DeltaFactor({"chi_1": 0.0, "chi_2": 0.3})
    samples, lp = delta.sample_and_log_prob(4, _DeviceContext(torch.device("meta")))
    assert all(v.device.type == "meta" for v in samples.values())
    assert lp.device.type == "meta"
    samples_cpu, lp_cpu = delta.sample_and_log_prob(4, context=None)
    assert lp_cpu.device.type == "cpu"  # no context device -> torch default


_HAS_MPS = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()


@pytest.mark.skipif(not _HAS_MPS, reason="needs an accelerator (MPS) for mixed devices")
def test_chain_device_consistency_on_accelerator():
    # A chain mixing a device-emitting factor, an (in-place) reparam, a correction,
    # and a DeltaFactor filler must stay on one device end to end -- the GPU pipe
    # case: constants follow the context device, transforms follow their inputs.
    device = torch.device("mps")

    class _DeviceGauss(_GaussFactor):
        def sample_and_log_prob(self, n, context, given=None):
            block, lp = super().sample_and_log_prob(n, context, given)
            return {k: v.to(device) for k, v in block.items()}, lp.to(device)

    comp = ChainComposer(
        [
            _DeviceGauss("x"),
            _InPlaceReparam(shift=5.0),
            _MockTargetCorrection(reads="x", emits="corr", consume=False),
            DeltaFactor({"c": 1.5}),
        ]
    )
    out = comp.sample(8, context=_DeviceContext(device))
    assert out["log_prob"].device.type == "mps"
    assert out["c"].device.type == "mps"
    assert out["x"].device.type == "mps"


class _FakeUncondModel:
    """A fake unconditional (density-recovery) model: a deterministic 'flow' whose
    draws are arange-based, with its OWN standardization distinct from the decoy
    base-model standardization under metadata["base"] -- the factor must read the
    model's own metadata."""

    def __init__(self):
        from types import SimpleNamespace

        self.metadata = {
            "train_settings": {
                "data": {
                    "unconditional": True,
                    "inference_parameters": ["H1_time_proxy"],
                    "standardization": {
                        "mean": {"H1_time_proxy": 2.0},
                        "std": {"H1_time_proxy": 0.5},
                    },
                }
            },
            # Decoy: using the base model's settings here would be a bug.
            "base": {
                "train_settings": {
                    "data": {
                        "inference_parameters": ["x"],
                        "standardization": {"mean": {"x": 0.0}, "std": {"x": 1.0}},
                    }
                }
            },
        }
        self.network = SimpleNamespace(eval=lambda: None)

    def sample_and_log_prob(self, num_samples=None):
        z = torch.arange(num_samples, dtype=torch.float32).unsqueeze(-1)
        lp = torch.full((num_samples,), -1.0)
        return z, lp

    def log_prob(self, z):
        return torch.full((z.shape[0],), -1.0)


def test_unconditional_flow_factor():
    # A density-recovery NDE takes no input: the factor never touches the context
    # (context=None), reads the model's own standardization and parameters, and
    # serves as a chain root (e.g. the single-step GNPE proxy source).
    factor = FlowFactor.from_model(_FakeUncondModel())
    assert factor.unconditional
    assert factor.parameters == ["H1_time_proxy"]
    assert factor.conditioning == []

    samples, lp = factor.sample_and_log_prob(4, context=None)
    # z = arange(4), destandardized with the NDE's own mean/std: z * 0.5 + 2.0.
    expected = torch.arange(4, dtype=torch.float32) * 0.5 + 2.0
    assert torch.equal(samples["H1_time_proxy"], expected)
    # Physical-space density: base lp plus the NDE's own -sum(log std).
    assert torch.allclose(lp, torch.full((4,), -1.0 - math.log(0.5)))

    # Re-plug through the no-context path.
    lp2 = factor.log_prob(samples, context=None)
    assert torch.allclose(lp2, lp)

    # Composes as a chain root with a conditioned factor downstream.
    comp = ChainComposer([factor, _ConstFactor("y", conditioning=["H1_time_proxy"])])
    out = comp.sample(3, context=None)
    assert out["y"].shape == (3,)
