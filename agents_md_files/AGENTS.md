# AGENTS.md

## Architecture

### Key Data Flow

1. **Training**: WaveformDataset + ASDDataset → train posterior model → `model.pt`
2. **Inference**: EventDataset (real/simulated data) → GWSampler → Result (posterior samples)
3. **Validation**: Result → sample missing parameters (phase) → importance sampling → reweighted samples + evidence

### Core Package Structure

**`dingo/core/`** — Neural network and inference foundations:

- `posterior_models/` — Flow-based posterior estimators (normalizing flows, flow matching, score diffusion)
- `nn/` — Neural network components: `enets.py` (embedding networks for GW data compression), `nsf.py` (neural spline flows), `cfnets.py` (coupling flows)
- `dataset.py` — Base DingoDataset with HDF5 serialization
- `result.py` — Posterior samples storage with importance sampling support

**`dingo/gw/`** — Gravitational wave-specific implementation:

- `likelihood.py` — StationaryGaussianGWLikelihood with time/phase/calibration marginalization
- `domains/` — Frequency/time domain definitions; `multibanded_frequency_domain.py` for frequency masking
- `waveform_generator/` — LALSuite waveform generation
- `transforms/` — Data transformation pipelines for waveforms, detectors, parameters. *Anytime you want to modify the training or inference loop, try doing it via transforms.*
- `training/` — Model training with `train_pipeline.py`
- `inference/` — GWSampler for posterior sampling

**`dingo/pipe/`** — Command-line pipeline (bilby_pipe-based):

- Four-stage workflow: data generation → sampling → importance sampling → plotting
- `parser.py` — INI file configuration parsing
- `nodes/` — HTCondor DAG node definitions

**`dingo/asimov/`** — Integration with Asimov, an LVK tool for automating analyses.

## Code Standards

### Code planning

- Think about creating tests before implementing them.

### Code writing

Before adding code, stop at the first option that solves the task correctly:

1. Do not build behavior that is not needed.
2. Use the standard library when it already solves the problem.
3. Use an existing repository or platform feature.
4. Use an already-installed dependency.
5. Inline a one-line solution when it remains readable; do not wrap it in a one-line helper.
6. Otherwise, write the minimum task-specific code that works.

General principles:

- No speculative abstractions, dependencies, classes, or boilerplate. Prefer deletion over addition and boring code over clever code.
- Question whether a complex requested mechanism is necessary when a simpler existing mechanism appears to cover the actual goal. Explain the alternative before implementing a materially larger design.
- When two approaches are similarly small, choose the one with correct edge-case behavior rather than the more fragile shortcut.
- This economy does not apply to validation at trust boundaries, errors that prevent silent data corruption, security, accessibility, scientific correctness, or explicitly requested hardware calibration.
- Non-trivial new logic must leave one small runnable check that would fail if the behavior breaks. Prefer the smallest focused test or assert-based self-check; trivial inline expressions need no dedicated test.
- Prefer the existing Dingo/JPNPE style: clean, readable, and compact.
- Prefer f-strings for string formatting.
- Prefer short, task-specific code over overly general abstractions. It is fine to refactor later if a script or approach becomes important.
- Before writing new code, check whether an appropriate function, class, or pattern already exists in the repo or in Dingo, and reuse it when possible.
- Strictly avoid unnecessary helper functions, especially one-line helpers or short helpers that are only used once. Inline the code unless extracting it makes the surrounding logic clearly easier to read.
- In long functions, use brief section-headline comments to mark the main phases of the function.
- Avoid adding new classes unless the user asks for one. If a class seems clearly necessary, ask first.
- When adding or modifying functions, include a short docstring that explains what the function does.
- New scripts and executable modules should start with a short module docstring explaining what the file does, especially when they contain a `main()` function.

## Environment Setup

- The user often already has a working environment on their machine, so do not use the default Python. Instead, ask the user where their environment is, and if it is not already recorded, put it into your memory.

## Things You Should Do

- Run smoke tests.

## Things You Should Not Do

- Do not modify the user's environment unless explicit approval is given.
- Do not add dependencies to the project unless approved by the user.
- Do not call functions from `gw/` into `core/`.

## Scientific Conventions
- phase 
- 