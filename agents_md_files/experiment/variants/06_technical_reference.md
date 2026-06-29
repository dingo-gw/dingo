# CLAUDE.md

## Project Overview
Dingo (Deep INference for Gravitational-wave Observations) performs Bayesian parameter
estimation of compact-binary mergers using neural posterior estimation (NPE) and its
group-equivariant variant (GNPE). Models are trained on simulated waveforms + detector noise
ASDs and then used to sample posteriors for real or simulated events, optionally reweighted by
importance sampling.

## Development Commands
- **Package management (uv recommended):** `uv sync`, `uv run <cmd>`.
- **Testing:** `uv run pytest`; single test `uv run pytest tests/.../test_x.py::test_name`; `-m "not slow"` to skip slow tests.
- **Formatting:** `uv run black dingo tests`.

## Architecture
- **`dingo/core/`** — model-agnostic ML/inference: `posterior_models/`, `nn/` (`enets.py`, `nsf.py`, `cfnets.py`), `dataset.py` (HDF5), `result.py` (samples + importance sampling).
- **`dingo/gw/`** — GW-specific: `likelihood.py`, `domains/` (incl. `multibanded_frequency_domain.py`), `waveform_generator/`, `transforms/`, `training/`, `inference/`.
- **`dingo/pipe/`** — CLI pipeline (bilby_pipe-based): generation → sampling → importance sampling → plotting; `parser.py`, `main.py`, `nodes/`.
- **CLI entry points:** `dingo_train`, `dingo_generate_dataset`, `dingo_pipe`, `dingo_ls`.

### Key Data Flow
1. **Training:** WaveformDataset + ASDDataset → train posterior model → `model.pt`.
2. **Inference:** EventDataset → GWSampler → Result (posterior samples).
3. **Validation:** Result → sample missing params (phase) → importance sampling → reweighted samples + evidence.

## Key Concepts
- **GNPE (Group-equivariant NPE):** exploits time-translation symmetry across detectors via Gaussian proxies; requires iterative sampling and trades exact `log_prob` access for accuracy. Initialized with an NPE network.
- **Training stages:** multi-stage schedule; later stages may freeze the SVD/embedding (reduced-basis) layer and change batch size / LR / ASD dataset.
- **Importance sampling:** reweights NPE/GNPE samples against the true likelihood to give unbiased posteriors + evidence; diagnostic is the effective sample size.
- **Multibanded frequency domain (MBFD):** compresses data by using coarser frequency resolution where the waveform varies slowly; a `base_domain` plus banding nodes. Watch for `UniformFrequencyDomain` vs `MultibandedFrequencyDomain` assumptions.

## Configuration
- Training/dataset/inference settings are plain nested dicts serialized into models/datasets (HDF5 / `.pt`), not dataclasses. They must round-trip losslessly.
- Model metadata lives under `metadata` in the checkpoint, with `dataset_settings` and `train_settings` subtrees.

## Practical Notes
- **Model types:** NPE (fast, exact `log_prob`) vs GNPE (more accurate for time-of-arrival, iterative, no direct `log_prob`).
- **Performance bottlenecks:** waveform generation and data conditioning dominate dataset generation; embedding-network SVD compression is key to manageable input sizes.
- **Conventions:** reuse existing transforms; modify training/inference behavior via `transforms/` where possible. Keep `core/` domain-agnostic.

## Boundaries
- Don't add dependencies or change the environment without approval. Don't commit/push.
