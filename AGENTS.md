# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Cursor, Gemini CLI, Copilot, etc.)
when working with code in this repository.

## What this is

Dingo (Deep Inference for Gravitational-wave Observations) performs gravitational-wave parameter
estimation via **neural posterior estimation**: a neural network (normalizing flow, flow matching, or
score matching) is trained to represent the Bayesian posterior $p(\theta|d)$ over source parameters
$\theta$ given detector data $d$. Training is expensive (~a week for production models), but inference
on new data is amortized and takes seconds.

## Environment & commands

Development uses [`uv`](https://docs.astral.sh/uv/). Install with `uv sync --extra dev` (add
`--extra wandb` / `--extra pyseobnr` for optional features).

```sh
uv run pytest tests/                              # full suite (asimov tests excluded by default)
uv run pytest tests/gw/test_injection.py          # single file
uv run pytest tests/gw/test_injection.py::test_x  # single test
uv run pytest -m asimov tests/                     # only the Asimov/LIGO pipeline integration tests
uv run pytest -m slow tests/                       # only slow-marked tests
uv run black dingo tests                           # format (black is the formatter)
```

- `addopts = "-m 'not asimov'"` in `pyproject.toml` means plain `pytest` skips `asimov`-marked tests;
  CI runs them as a separate step. Two custom markers exist: `slow` and `asimov`.
- CI (`.github/workflows/pytest.yml`) runs the suite on Python 3.10–3.13 via `uv sync --extra dev`.
- `ci/` is a *separate* CI mechanism: a standalone bash + Docker + systemd setup that runs the
  toy-NPE example end-to-end on new commits/tags — unrelated to the GitHub Actions pytest workflow.

## Architecture

### `core` vs `gw` split (Bilby convention)

Code is partitioned so it can extend beyond LVK gravitational waves:
- `dingo/core/` — generic, domain-agnostic: neural network architectures (`nn/`), posterior model
  classes (`posterior_models/`), samplers, the `DingoDataset` base class, generic transforms,
  density estimation (`density/`).
- `dingo/gw/` — GW-specific: waveform generation, detector projection, domains, priors, GW datasets,
  GW samplers, importance sampling.

When generalizing code, the intended direction is migrating things from `gw` up into `core`.

### Settings-driven, reproducibility-first

This is the central design principle. Every task is configured by a **nested-dictionary settings file**
(YAML for most commands; INI for `dingo_pipe`, matching LVK convention) passed to a `dingo_*`
command-line script. Each script is a thin wrapper around a library function: it reads the settings,
runs, and writes an output file (`.hdf5` for data, `.pt` for trained networks).

**Output files embed their own settings — including settings inherited from precursor steps.** A trained
model `.pt` carries the waveform-dataset and ASD-dataset settings that produced its training data, not
just the training settings. Downstream tasks rely on these embedded precursor settings (e.g. combining
the intrinsic prior from the waveform dataset with the extrinsic prior used at training). Inspect any
Dingo output file with `dingo_ls`. When adding configuration, prefer extending the settings dict
(backward-compatible) over changing function signatures.

### Transforms pipeline

Data preprocessing is a sequence of composable transforms (PyTorch style) in `dingo/{core,gw}/transforms/`:
sample extrinsic parameters, project polarizations to detectors, add noise, etc. **The same transform
sequence is reused at inference time** (similar but not identical to training); transforms that must
behave differently carry an inference-mode flag. If you change a training transform, check the
corresponding inference path (`gw/transforms/inference_transforms.py`).

### Datasets

All dataset classes inherit `dingo.core.dataset.DingoDataset`, which provides HDF5/dict IO and stores
the settings dict as an attribute. Waveform datasets are often SVD-compressed.

### Posterior model types

`dingo/core/posterior_models/` provides interchangeable model backends behind `build_model.py`:
normalizing flow (NPE), flow matching (FMPE), and score matching — all sharing `base_model.py`. Example
configs for each live under `examples/` (`npe_model/`, `fmpe_model/`, `gnpe_model/`, `toy_npe_model/`).

### GNPE (two-network workflow)

GNPE combines physical symmetries with Gibbs sampling for better results. It requires training **two**
networks: the main conditional network plus a smaller initialization network that seeds the Gibbs
sampler (estimates per-detector coalescence times). Inference must be pointed at both. Enabled via the
`data/gnpe_time_shifts` option in training settings. See `docs/source/gnpe.md`.

## Typical end-to-end workflow

1. `dingo_generate_dataset` → waveform polarization dataset (HDF5). Cluster version:
   `dingo_generate_dataset_dag`.
2. `dingo_generate_asd_dataset` → detector noise ASD dataset (HDF5).
3. `dingo_train --settings_file train_settings.yaml --train_dir <dir>` → trained `.pt` (resume with
   `--checkpoint`). Cluster version: `dingo_train_condor`. Select GPU via `CUDA_VISIBLE_DEVICES`.
4. `dingo_pipe` → inference on real/simulated data from an INI file (HTCondor-compatible).
5. Importance sampling corrects/verifies results (`dingo/gw/importance_sampling/`,
   `dingo_pipe_importance_sampling`).

All `dingo_*` console-script entry points are declared in `pyproject.toml` under `[project.scripts]` —
that table is the authoritative map from command name to the function it wraps.

`dingo/pipe/` is the bilby_pipe-style orchestration layer for inference (DAG creation, data generation,
sampling, plotting, PP tests). `dingo/asimov/` integrates Dingo as an Asimov pipeline (registered via
the `asimov.pipelines` entry point).

## Docs

Source under `docs/source/` (Sphinx + MyST). `code_design.md` and `overview.md` are the best
architectural references; per-task guides exist for training, inference, GNPE, importance sampling, and
each example model.
