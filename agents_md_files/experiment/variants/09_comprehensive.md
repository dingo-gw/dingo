# AGENTS.md — DINGO

> Deep INference for Gravitational-wave Observations. Bayesian parameter estimation of
> compact-binary coalescences via neural posterior estimation (NPE / GNPE), with optional
> importance sampling for unbiased posteriors and evidences. This file is the single source of
> truth for working in this repository; keep changes consistent with it.

---

## 1. Quick start / commands

```bash
# Environment (uv is the supported workflow; do NOT use a system Python)
uv sync                                      # create/sync the virtualenv from uv.lock
uv run python -c "import dingo; print(dingo.__file__)"

# Tests
uv run pytest                                # full suite
uv run pytest tests/gw/test_x.py::test_name  # a single test (use this while iterating)
uv run pytest -m "not slow"                  # skip slow-marked tests
uv run pytest -m "not asimov"                # skip asimov-marked tests

# Formatting / lint
uv run black dingo tests                     # canonical formatter; run before finishing

# CLI entry points
dingo_generate_dataset   # build a waveform/ASD dataset from a settings file
dingo_train              # train a posterior model -> model.pt
dingo_pipe <config.ini>  # full inference pipeline (DAG): generation -> sampling -> IS -> plotting
dingo_ls <file>          # inspect metadata of an HDF5 dataset or a .pt checkpoint
```

Pin BLAS threads to match CI when running numerical tests:
`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.

---

## 2. Architecture map

Three layers, strictly ordered (upper may import lower, never the reverse):

### `dingo/core/` — model-agnostic ML & inference
- `posterior_models/` — flow-based posterior estimators: normalizing flows, flow matching, score-based diffusion. Common `Base` interface with `.sample()`, `.log_prob()` (where defined), and `metadata`.
- `nn/` — building blocks: `enets.py` (embedding networks that compress GW data, incl. SVD reduced-basis layer), `nsf.py` (neural spline flows), `cfnets.py` (continuous/coupling flows).
- `dataset.py` — `DingoDataset` base with recursive HDF5 serialization.
- `result.py` — posterior-sample container with importance-sampling support (weights, ESS, evidence).
- **Rule:** nothing here may import GW/bilby/LAL/astropy-GW code. Keep it domain-agnostic.

### `dingo/gw/` — gravitational-wave specifics
- `likelihood.py` — `StationaryGaussianGWLikelihood` with time/phase/calibration marginalization.
- `domains/` — frequency/time domain definitions; `build_domain.py` (`build_domain_from_model_metadata`); `multibanded_frequency_domain.py` (MBFD masking/compression).
- `waveform_generator/` — LALSuite / pyseobnr waveform generation.
- `transforms/` — data transformation pipelines (waveforms, detectors, parameters). **Prefer to express changes to the training or inference loop as transforms.**
- `training/` — `train_pipeline.py`, `train_builders.py` (assemble model + data from settings).
- `inference/` — `GWSampler` / `GNPESampler` for posterior sampling.

### `dingo/pipe/` — command-line pipeline (bilby_pipe-based)
- Four stages: **data generation → sampling → importance sampling → plotting**, run as an HTCondor DAG.
- `parser.py` — declares every INI argument. New config options go here.
- `main.py` — `dingo_pipe` entry: `fill_in_arguments_from_model()` reads the model **metadata** to fill derived args; `write_complete_config_file()` writes `{outdir}/{label}_config_complete.ini`; `generate_dag()` builds the DAG.
- `nodes/` — per-stage HTCondor node definitions; each declares the files it transfers to compute nodes.
- `data_generation.py`, `plot.py`, `sampling.py` — the stage implementations.

### `dingo/asimov/` — integration with Asimov, the LVK analysis-automation tool.

---

## 3. Key data flow

1. **Training:** `WaveformDataset` + `ASDDataset` → train posterior model → `model.pt`.
2. **Inference:** `EventDataset` (real or simulated) → `GWSampler` → `Result` (posterior samples).
3. **Validation:** `Result` → sample missing parameters (e.g. phase) → importance sampling → reweighted samples + evidence.

---

## 4. Key concepts (domain glossary)

- **NPE** — Neural Posterior Estimation: a conditional flow `q(θ|data)` trained by maximum likelihood; gives fast samples and exact `log_prob`.
- **GNPE** — Group-equivariant NPE: exploits detector time-translation symmetry using Gaussian proxies; more accurate for times-of-arrival but requires **iterative** sampling and gives up direct `log_prob`. A GNPE setup is *initialized with an NPE network*.
- **Importance sampling (IS):** reweight NPE/GNPE samples by the ratio of the true likelihood×prior to the network density → unbiased posterior and evidence. Effective sample size (ESS) is the headline diagnostic.
- **Multibanded frequency domain (MBFD):** compresses strain by using coarser Δf where the waveform is slowly varying; defined by a `base_domain` plus banding `nodes`. Many bugs come from confusing `UniformFrequencyDomain` and `MultibandedFrequencyDomain` — always confirm which one a code path assumes.
- **Reduced basis / SVD:** the embedding network projects high-dimensional frequency data onto an SVD basis to keep inputs tractable.

---

## 5. Configuration & metadata conventions

- Settings (dataset, training, inference) are **plain nested dicts**, serialized into datasets/models (HDF5, `.pt`). They are **not** dataclasses; do not introduce dataclass conversions. They must round-trip losslessly.
- A checkpoint (`model.pt`) holds `model_kwargs`, `model_state_dict`, `epoch`, `version`, and **`metadata`**. The `metadata` has two subtrees:
  - `dataset_settings` — `domain` (incl. `base_domain` f_min/f_max/delta_f), `intrinsic_prior`, `waveform_generator` (approximant, f_ref, …), `compression.svd`, `num_samples`.
  - `train_settings` — `data` (detectors, ref_time, extrinsic_prior, standardization mean/std), `model` (embedding + posterior kwargs), `training` (per-stage schedule).
- The pipeline's **completed config** (`*_config_complete.ini`) is the artifact passed between DAG nodes. Prefer carrying information through it rather than re-reading large artifacts on each node.

---

## 6. Code standards

### Decision ladder — stop at the first option that solves the task correctly
1. Don't build behavior that isn't needed.
2. Use the standard library when it already solves the problem.
3. Use an existing repository or platform feature.
4. Use an already-installed dependency.
5. Inline a one-line solution when it stays readable; don't wrap it in a one-line helper.
6. Otherwise, write the minimum task-specific code that works.

### General principles
- No speculative abstractions, dependencies, classes, or boilerplate. Prefer deletion over addition; boring code over clever code.
- Question whether a complex requested mechanism is necessary when a simpler existing one covers the goal; explain the alternative before implementing a materially larger design.
- When two approaches are similarly small, pick the one with correct edge-case behavior, not the more fragile shortcut.
- The economy above does **not** apply to: validation at trust boundaries, errors that prevent silent data corruption, security, accessibility, or **scientific correctness**.
- Reuse first — search the repo for an existing function/class/transform/pattern before writing new code.
- Strictly avoid one-line or single-use helper functions; inline unless extraction clearly improves readability.
- Prefer f-strings. Use brief section-headline comments to mark phases of long functions.
- Add a short NumPy-style docstring to new/modified functions. Start new executable modules with a module docstring.
- PEP 604 type hints (`X | None`). Avoid adding new classes unless asked; if one seems necessary, ask first.

### Style examples
```python
# Good: reuse existing builder, smallest change, typed, documented.
def build_domain_from_settings(settings: dict) -> Domain:
    """Build the (possibly multibanded) frequency domain from dataset settings."""
    return build_domain(settings["domain"])

# Avoid: a single-use one-line helper wrapping a trivial call.
def _get_domain(s):  # noqa - don't do this
    return build_domain(s["domain"])
```

---

## 7. Testing

- Think about the test **before** implementing. Non-trivial new logic must leave a small runnable check that fails if the behavior breaks.
- Tests are deterministic: seed RNG, pin thread counts (see §1). Use existing fixtures/conventions.
- Run a single test with `uv run pytest <path>::<name>`; only run the full/slow suite when asked.
- An end-to-end **smoke test** runs the toy pipeline (small dataset → brief train → sample) to confirm the pipeline still works; use it for integration sanity, not unit coverage.

---

## 8. Git & workflow

- Read-only git is fine (`status`, `diff`, `log`, `show`). Do **not** stage, commit, push, branch, or rebase unless explicitly asked — leave changes in the working tree for review.
- Keep commits (when requested) focused; describe the *why*.

---

## 9. Do / Ask-first / Never

**Always**
- Format with `uv run black` before finishing.
- Reuse existing code; add a focused test for new logic; keep `core/` domain-agnostic.

**Ask first**
- Adding a dependency or changing the environment.
- Introducing a new class, a new config schema, or a materially larger design than requested.
- Changing serialization shapes (HDF5/`.pt` metadata) that affect existing artifacts.

**Never**
- Modify the user's environment or install packages without approval.
- Call from `gw/` into `core/`, or add GW/bilby/LAL imports to `core/`.
- Commit datasets, checkpoints, or debug `print`s; add a logging framework.
- Commit/push without being asked.

---

## 10. Gotchas

- `UniformFrequencyDomain` vs `MultibandedFrequencyDomain` — verify the assumed domain on any frequency-indexed code path.
- Geocent-time conventions differ between Dingo (centered near 0) and Bilby (centered at trigger time); the pipeline rebuilds the `geocent_time` prior accordingly.
- Waveform generation / data conditioning dominate dataset-generation time; SVD compression keeps embedding inputs tractable.
- pyseobnr may fail to import in some environments (SEOBNRv5 approximants unavailable); don't treat that as your bug unless the task is about it.

---

## 11. Context / compaction note

When compacting a long session, preserve: the list of files modified, the exact test command(s) being used, and any scientific assumptions made (priors, domains, conventions). Those are easy to lose and expensive to reconstruct.
