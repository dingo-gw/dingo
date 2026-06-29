# DINGO — Codebase Review & Hackathon Improvement Map

*Prepared as preparation for the Dingo hackathon. Focus: maintainability, code
quality, and developer experience — not the correctness of the underlying
physics/ML, which is sound.*

## 1. What Dingo is (orientation)

Dingo is a neural posterior estimation framework for gravitational-wave
inference: it trains normalizing flows / flow-matching / score-diffusion models
to represent Bayesian posteriors conditioned on detector data, then corrects
them with importance sampling. ~24.6k LOC of Python across 121 files in three
layers:

- **`dingo/core/`** — model-agnostic NN + inference machinery (posterior
  models, embedding nets, samplers, `Result`, HDF5 dataset I/O).
- **`dingo/gw/`** — GW-specific physics (waveform generation, likelihood,
  transforms, training, inference, noise/ASD).
- **`dingo/pipe/`** — bilby_pipe-based CLI/DAG orchestration for end-to-end runs.

## 2. Overall verdict

This is a **scientifically mature, well-documented research codebase** that has
accreted complexity as it grew from papers into a production LVK tool. The
*physics and ML are sound and well-explained*; the *software-engineering hygiene
has not kept pace*. The highest-leverage hackathon work is **not** rewriting
algorithms — it's **enforcement tooling, test coverage of user-facing paths,
decomposing a handful of god-files, and replacing 237 `print()`s with logging.**
Most of this is low-risk and parallelizable across hackathon teams.

## 3. What works well

- **Documentation & docstrings** are genuinely strong — math derivations inline
  (likelihood, evidence, ODE formulations), 21 doc files, 4 Jupyter tutorials,
  ReadTheDocs configured, a `code_design.md` explaining the core/gw split.
- **Clean core abstractions**: `BasePosteriorModel` ABC with a tidy
  `build_model_from_kwargs` factory; `NPE → ContinuousFlow → {FlowMatching,
  ScoreDiffusion}` hierarchy is sensible; the transform `Compose` / sample-dict
  convention is a good design.
- **Reproducibility discipline**: everything (settings, metadata) round-trips
  through HDF5; outputs are self-describing.
- **Modern packaging**: `pyproject.toml` + `uv.lock` + `setuptools_scm`,
  dependency groups for dev/docs/typing, 32 well-named CLI entry points, PyPI +
  conda-forge distribution.
- **Test foundation exists and is decent quality** where present: fixtures,
  parametrization, `slow`/`asimov` marks, numerical-correctness assertions in
  `core/` and `gw/transforms/`.

## 4. Top problems (cross-cutting, verified)

| # | Problem | Evidence | Severity |
|---|---------|----------|----------|
| 1 | **No quality enforcement in CI.** `black` is a dev dep but not enforced; no `ruff`/`flake8`, no `mypy`, no `pre-commit`, no coverage. CI only runs pytest. | No `.pre-commit-config.yaml`/`ruff`/`mypy` configs exist; `.github/workflows/pytest.yml` runs tests only | **High** |
| 2 | **`print()` everywhere instead of logging.** 237 `print()` calls vs only 3 files using `logging` — and a `logging_utils.py` already exists, unused. Can't control verbosity in distributed/condor jobs. | `dingo/core/utils/logging_utils.py` present but bypassed | **High** |
| 3 | **God-files needing decomposition.** `waveform_generator.py` (1641 LOC, two classes ~80% duplicated), `pipe/parser.py` (1608 LOC, ~176 `add()` calls + ~200 lines commented-out), `core/result.py` (1114), `likelihood.py` (885, 4 near-duplicate `_log_likelihood_*` methods). | file sizes; `# TODO: Combine with _log_likelihood()` at likelihood.py:622 | **High** |
| 4 | **Test coverage gaps on user-facing paths.** Entire `dingo/pipe/`, `gw/training/`, `gw/inference/`, `gw/importance_sampling/`, and `core/samplers.py` have **no tests**. Test/code ratio ≈16%. No coverage measured in CI. | tests/ tree vs source tree | **High** |
| 5 | **Layering violation: core depends on gw.** `core/samplers.py:19` imports `from dingo.gw.transforms import …`, breaking the core/gw separation the docs advertise. | verified | **Med** |
| 6 | **Latent bug in config comparison.** `pipe/main.py:116`: `if isinstance(args_v, str) and "{" and "}" in args_v:` — the `and "{"` is a constant-truthy no-op; intent was `"{" in args_v and "}" in args_v`. | verified | **Med** |
| 7 | **Fragile bilby_pipe coupling.** Version-gated API swap (`parser.py:67–73`) + a monkey-patch (`data_generation.py:249`). bilby_pipe updates routinely break this, yet none of it is tested. | verified | **Med** |
| 8 | **Bare `except:` (7×) and silent NaN fallbacks** hide real failures (e.g. `likelihood.py:870`, `condor_utils.py:78`). | grep: 7 bare excepts | **Med** |
| 9 | **Committed binary test artifact.** `tests/gw/transforms/waveform_data.npy` is tracked despite `.gitignore` excluding `*.npy`. | `git ls-files` | **Low** |
| 10 | **Repo-root clutter:** `misc_scripts/` (14 undocumented scripts), `compatibility/`, `ci/` (a bespoke Docker/systemd toy-model runner) sit at top level with unclear status; 56 TODO/FIXME markers. | listing | **Low** |
| 11 | **Sparse type hints** (only 19/121 files have return annotations) despite typing-stub deps being declared. | grep | **Low** |

## 5. Recommended hackathon plan

These are scoped so independent teams can work in parallel with minimal merge
conflict.

**Tier 1 — high impact, low risk (do these first):**

1. **Add `pre-commit` + CI enforcement**: `black --check`, `ruff`, and `mypy`
   (start lenient) as a separate CI job. This *locks in* every later cleanup.
   Add `pytest-cov` with a low initial floor.
2. **`print()` → `logging`** sweep using the existing `logging_utils.py`.
   Mechanical, isolated per-module, instantly reviewable.
3. **Repo hygiene**: untrack `waveform_data.npy` (regenerate via fixture or
   Git-LFS), delete the ~200 lines of commented-out parser code, triage the 56
   TODOs into GitHub issues, add a `CONTRIBUTING.md`.

**Tier 2 — medium effort, high value:**

4. **Characterization tests for `dingo/pipe`** (parser round-trips,
   `fill_in_arguments_from_model`) — this both fixes bug #6 and creates the
   safety net for the brittle bilby_pipe coupling.
5. **Decompose `pipe/parser.py`** into per-domain argument modules
   (calibration/data-gen/sampler) behind a factory.
6. **Fix the core→gw import** via dependency injection (pass the transform into
   `Sampler`).

**Tier 3 — larger refactors (scope as stretch goals):**

7. **De-duplicate `waveform_generator.py`** — extract a shared
   parameter-converter / `WaveformBackend` abstraction so
   `NewInterfaceWaveformGenerator` reuses rather than re-implements (~80%
   overlap today).
8. **Consolidate the 4 `_log_likelihood_*` methods** in `likelihood.py` (the
   code already has a TODO requesting this).

## 6. Subsystem notes

### dingo/gw — physics core (waveform generation, likelihood, transforms)

- **Works well**: strong docstrings with math; transform pipeline pattern;
  graceful physics-domain error handling (NaN + warning on waveform failure);
  clean GWSignal/Injection/Likelihood separation.
- **Problems**: `waveform_generator.py` two classes duplicate `_convert_parameters`
  (231-line god-function), `generate_hplus_hcross_m`, and FD/TD mode generators;
  tight coupling to LAL/gwsignal internals with no backend abstraction; 4
  near-duplicate `_log_likelihood_*` methods; magic numbers (`l_max = 5`,
  hardcoded thresholds, ODE tolerances); bare `except:` returning NaN.

### dingo/core — model-agnostic ML/inference

- **Works well**: `BasePosteriorModel` ABC + factory; clean continuous-flow
  hierarchy; robust importance-sampling math (logsumexp); generic recursive
  HDF5 I/O with partial-load indexing.
- **Problems**: `print()` instead of the existing logger; **core→gw import**
  (layering violation) in `samplers.py:19`; `Result.importance_sample()`
  god-method (~89 lines mixing validation/likelihood/weights/evidence);
  duplicated `odeint` blocks in `cflow_base.py` with repeated `atol/rtol=1e-7`;
  fragile nested-dict config access (`metadata["train_settings"]["model"]`).

### dingo/pipe & dingo/gw/training — CLI / orchestration

- **Works well**: centralized `dingo_pipe` logger in most paths; modular DAG
  node structure; good type hints in the training builders.
- **Problems**: `parser.py` monolith (1608 lines, ~200 commented-out); `MainInput`
  god-class (~160 attributes); broken `isinstance` check (bug #6); fragile
  bilby_pipe version detection + monkey-patch; 25+ `print()` in training code;
  **no unit tests** for parser/main; unexplained default hyperparameters.

### Tests, docs, examples, tooling

- **Tests**: good unit/integration mix in `core/` and `gw/transforms/`, but
  ~55% of the codebase (all of `pipe/`, `training/`, `inference/`,
  `importance_sampling/`, `core/samplers.py`) is untested; no coverage in CI; a
  257 KB `.npy` is committed; no root `conftest.py`; optional deps (bilby/LAL)
  not guarded with `skipif`.
- **Docs**: comprehensive (21 files + 4 notebooks), ReadTheDocs configured. Gaps:
  no `CONTRIBUTING.md`, no clearly-surfaced API reference, no troubleshooting/FAQ,
  examples reference external datasets without version/date stamps.
- **Tooling**: no `pre-commit`, no lint/type checks in CI, `black` not enforced;
  `ci/`, `compatibility/`, `misc_scripts/` clutter the repo root with unclear
  status.

## 7. Caveats on this review

The subsystem deep-dives were produced by exploration agents reading excerpts.
The most consequential and surprising claims were independently verified by
direct inspection: the `main.py:116` bug, the `core`→`gw` import, the committed
`.npy`, the absence of lint/type/pre-commit configs, and the print/except
counts. Line citations for those verified items are reliable; a few finer-grained
line numbers in the underlying analysis were not re-checked exhaustively — worth
confirming before acting on the smaller ones.
