# AGENTS.md — dingo

Guidance for AI coding agents working in this repo. This file is the single source of
truth for project knowledge; it is tool-agnostic (read by any AGENTS.md-aware agent).
Claude Code reads it via the `CLAUDE.md` symlink and layers extra automation on top in
`.claude/` (skills, subagent, hooks, settings) — but every fact needed to work here lives
in *this* file.

## What this is

**dingo** (Deep Inference for Gravitational-wave Observations) does neural posterior
estimation for gravitational-wave inference: build training datasets, train normalizing
flows / (continuous) flow- and score-matching models, sample posteriors, and verify with
importance sampling. The code is layered:

- `dingo/core/` — model-agnostic ML (posterior models, NN architectures, datasets,
  samplers, results). **Keep this domain-agnostic — no GW assumptions here.**
- `dingo/gw/` — gravitational-wave domain code (likelihood, priors, waveform generation,
  domains, transforms, training orchestration).
- `dingo/pipe/` — end-to-end analysis pipeline built on `bilby_pipe` (HTCondor DAGs).
- `dingo/asimov/` — LVK Asimov integration (the `Dingo` pipeline class + `dingo.ini`).

## Setup

Python `>=3.10,<3.14`. Dependencies are managed with **uv**.

```bash
uv sync                                          # dev environment (editable install)
uv sync --extra wandb --extra pyseobnr           # with optional extras
```

Prefix project commands with `uv run` so they use the project venv (e.g. `uv run pytest`).

## Common commands

```bash
# Tests (asimov tests are excluded by default via addopts = -m 'not asimov')
uv run pytest tests/
uv run pytest tests/gw/path/to/test_x.py::test_y   # single test
uv run pytest -m asimov tests/                      # opt-in: asimov integration tests
uv run pytest -m "not slow" tests/                  # skip slow tests

# Formatting (black; not enforced in CI, so run it yourself)
uv run black dingo tests

# Docs (Sphinx)
cd docs && uv run make html

# Inspect dingo output files (settings/metadata in .hdf5 / .pt)
dingo_ls <file>
```

Key CLI entry points (defined in `pyproject.toml`): `dingo_train`,
`dingo_generate_dataset`, `dingo_generate_asd_dataset`, `dingo_pipe`, `dingo_result`,
`dingo_ls`, plus `*_condor` / `*_dag` variants for cluster runs.

## Architecture map (where things live)

- Posterior models / ABC: `dingo/core/posterior_models/` (`base_model.py`,
  `normalizing_flow.py`, `flow_matching.py`, `score_matching.py`).
- NN layers: `dingo/core/nn/` (`nsf.py` neural spline flow, embedding/coupling nets).
- Datasets & HDF5 I/O: `dingo/core/dataset.py` (`DingoDataset`, recursive HDF5),
  `dingo/gw/dataset/waveform_dataset.py`.
- Results & importance sampling: `dingo/core/result.py`, `dingo/gw/result.py`.
- GW likelihood: `dingo/gw/likelihood.py` (large, central — phase/time marginalization).
- Waveforms: `dingo/gw/waveform_generator/waveform_generator.py` (LAL interface, large).
- Frequency domains: `dingo/gw/domains/` (`UniformFrequencyDomain` vs
  `MultibandedFrequencyDomain` — this choice affects likelihood & waveform code).
- Training loop: `dingo/gw/training/train_pipeline.py`, builders in `train_builders.py`.
- Pipeline / INI parsing: `dingo/pipe/main.py`, `dingo/pipe/parser.py` (extends bilby_pipe).

## Conventions

- **Docstrings:** NumPy style (Parameters / Returns sections); Sphinx napoleon renders them.
- **Type hints:** modern, PEP 604 unions (`X | Y`), Python 3.10+ syntax.
- **Formatting:** black defaults. Run `uv run black dingo tests` before finishing.
- **Config:** plain **nested dicts loaded from YAML/INI**, *not* dataclasses. Settings are
  embedded in output HDF5 metadata (inspect with `dingo_ls`) for reproducibility.
- **Persistence:** HDF5 I/O is recursive over nested dicts (`recursive_hdf5_load/save`).
- **Transforms:** composed via `torchvision.transforms.Compose` (not `nn.Sequential`).
- **Naming:** `PascalCase` classes, `snake_case` functions, `UPPER_SNAKE` constants,
  `_leading_underscore` for internals.
- **Logging:** sparse; the codebase mostly prints. Don't add a heavy logging framework.

## Testing rules

- `tests/conftest.py` has an autouse fixture seeding numpy & bilby RNGs before each test,
  so tests are deterministic regardless of order. **Keep new tests deterministic** — don't
  rely on unseeded randomness or wall-clock.
- Markers: `slow`, `asimov`. Asimov tests are excluded by default; run them explicitly.
- Thread-count env vars (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`) are pinned to `1` in CI to avoid oversubscription; set the same
  locally when running tests (Claude Code sets these automatically via `.claude/settings.json`).

## End-to-end smoke test (toy NPE)

The fastest way to confirm a change didn't break the pipeline. Run from a scratch copy of
`examples/toy_npe_model/` (don't write outputs into the repo). Mirrors `ci/dingo-ci`:

```bash
uv run dingo_generate_dataset --settings waveform_dataset_settings.yaml \
    --out_file training_data/waveform_dataset.hdf5
uv run dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml \
    --data_dir training_data/asd_dataset
uv run dingo_train --settings_file train_settings.yaml --train_dir training
uv run dingo_pipe GW150914.ini
```

## Gotchas

- `UniformFrequencyDomain` vs `MultibandedFrequencyDomain` is load-bearing — check which
  one a code path assumes before changing likelihood/waveform logic.
- bilby conventions are inherited deeply (priors, interferometers, parameter names).
- Asimov integration is tightly coupled to the installed `asimov` package version.
- Phase/time marginalization in the likelihood is optional and changes the computation.

## Do / Don't

- **Do** keep `core/` domain-agnostic; put GW specifics in `gw/`.
- **Do** run `uv run black` and the relevant `uv run pytest` subset before finishing.
- **Don't** commit generated datasets, ASDs, checkpoints, or training output dirs.
- **Don't** convert dict-based settings to dataclasses or restructure the config system
  without explicit agreement — it's intentional and tied to reproducibility/metadata.
