# Agent Instructions — dingo

Dingo does gravitational-wave parameter inference with neural posterior estimation.

## Environment and execution
- Use the project environment via `uv` (`uv sync`, `uv run ...`). Do **not** fall back to a system Python, and do not install or upgrade packages without explicit approval.
- Assume a GPU may be available but write code that also runs on CPU (`device` is configurable).
- Put scratch/debug artifacts in a repo-local throwaway dir (e.g. `scratch_runs/`), never in the user's home or `/tmp` mixed with other work. Don't commit them.

## Git
- Do not stage, commit, push, switch branches, or rebase. Leave the working tree changes for the user to review.
- Read-only git (`status`, `diff`, `log`, `show`) is fine.

## Implementation discipline
- Reuse first: search the repo (and dingo) for an existing function/class/pattern before writing new code.
- Prefer the smallest change that correctly solves the task. No speculative abstractions or one-line helpers.
- Keep `dingo/core/` domain-agnostic — no GW/bilby/LAL imports there.
- Add a short docstring to new/changed functions; start new executable modules with a module docstring.
- Use brief section-headline comments to mark phases of long functions.

## Testing and smoke tests
- Add or update a focused, deterministic test for new logic (RNG seeded; pin BLAS threads with `OPENBLAS_NUM_THREADS=1` etc. to match CI).
- Run a single test with `uv run pytest <path>::<name>`; avoid running the full slow suite unless asked.
- A quick end-to-end "smoke test" exercises the toy pipeline (dataset → short train → sample); use it to confirm you didn't break the pipeline, not for unit-level checks.

## Working from a task list
- Break the task into concrete steps and track status (Ready → In progress → Needs review).
- When a scientific choice is ambiguous (priors, domains, conventions), state the assumption and flag it for review rather than silently guessing.

## Architecture (orientation)
- `dingo/core/` model-agnostic ML; `dingo/gw/` GW physics (waveforms, likelihood, domains, transforms); `dingo/pipe/` CLI pipeline (generation → sampling → importance sampling → plotting); `dingo/asimov/` LVK automation.
