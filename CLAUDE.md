Dingo: gravitational-wave parameter inference using neural posterior estimation.

## Architecture
- **`dingo/core/`** — model-agnostic ML/inference: `posterior_models/` (normalizing flows, flow matching, score diffusion), `nn/` (embedding nets, neural spline flows), `dataset.py` (HDF5 serialization), `result.py` (posterior samples + importance sampling).
- **`dingo/gw/`** — GW-specific: `likelihood.py`, `domains/`, `waveform_generator/`, `transforms/`, `training/`, `inference/`.
- **`dingo/pipe/`** — CLI pipeline (bilby_pipe-based): data generation → sampling → importance sampling → plotting. `parser.py` (INI), `main.py` (completed config + DAG), `nodes/` (HTCondor).
- **`dingo/asimov/`** — Asimov (LVK) integration.

## Commands
- `uv sync`; `uv run pytest [tests/...::test]`; `uv run black dingo tests`.

## Code writing — stop at the first option that solves the task correctly
1. Don't build behavior that isn't needed.
2. Use the standard library when it already solves the problem.
3. Use an existing repository or platform feature.
4. Use an already-installed dependency.
5. Inline a one-line solution when it stays readable; don't wrap it in a one-line helper.
6. Otherwise, write the minimum task-specific code that works.

General principles:
- No speculative abstractions, dependencies, classes, or boilerplate. Prefer deletion over addition, boring code over clever code.
- When two approaches are similarly small, choose the one with correct edge-case behavior.
- This economy does **not** apply to validation at trust boundaries, errors preventing silent data corruption, security, or scientific correctness.
- Non-trivial new logic must leave one small runnable check that fails if the behavior breaks.
- Reuse existing functions/classes/patterns before writing new ones.
- Strictly avoid one-line / single-use helpers — inline unless extraction clearly improves readability.
- Prefer f-strings. Add a short docstring to new/modified functions. Start new executable modules with a module docstring.
- Avoid adding new classes unless asked; if one seems necessary, ask first.

## Testing
- Think about the test before implementing. Add a focused test or assert-based self-check for new logic.
- Tests are deterministic (RNG seeded); run a single test with `uv run pytest <path>::<name>`.

## Git
- Don't run interactive git, and don't commit or push unless asked.

## Boundaries
- Don't modify the user's environment or add dependencies without explicit approval.
- Don't call functions from `gw/` into `core/` (keep `core/` domain-agnostic).