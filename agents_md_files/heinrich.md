# Codex Instructions

## Environment And Execution

- Use the `tuebingen` micromamba environment for tests, smoke tests, and project
  scripts unless explicitly instructed otherwise:
  `micromamba run -n tuebingen ...`
- Assume a GPU is available unless told otherwise. GPU smoke tests should use the
  same environment.
- Stay inside the repository by default. Do not inspect or write files outside
  the repo unless the user explicitly asks for it or gives an external path to
  use for the current task.
- Put ad hoc smoke-test outputs, temporary configs, and one-off debugging scripts
  under `codex_debugging_runs/`.
- Keep smoke tests small: reduce epochs, batch sizes, steps, samples, and model
  sizes so the check finishes quickly while still exercising the relevant path.

## Git

- Do not touch git unless explicitly asked. Do not run git commands, create
  commits, reset files, checkout files, or otherwise manage version control.

## Hydra And Logging

- Prefer Hydra for scripts that need configurable experiments unless explicitly
  instructed otherwise. Keep each script's settings in one concise YAML file;
  do not introduce config groups or defaults lists unless they remove real
  duplication.
- Use the standard repository pattern:
  `@hydra.main(version_base=None, config_path=..., config_name=...)` and
  `log = logging.getLogger(__name__)`.
- Let Hydra configure logging. Do not add custom handlers, formatters, or
  `logging.basicConfig()` to Hydra scripts. Configure the output and log name in
  YAML so Hydra writes `<script_name>.log` in the job directory:

  ```yaml
  hydra:
    run:
      dir: ${outdir}
    job:
      name: script_name
  ```

- At startup, log the complete resolved configuration with
  `log.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")` and save it as
  `settings.yaml` in the job directory using
  `OmegaConf.save(cfg, path, resolve=True)`. Hydra's `.hydra/` snapshot may stay
  as additional provenance, but it does not replace the explicit resolved copy.
- For workflows that invoke several scripts, resolve and copy all input configs
  at the beginning of the workflow. Later stages must use those snapshots so a
  concurrent edit to a source YAML cannot change an already-running experiment.
- Non-Hydra scripts should use a few plain logging lines and log their complete
  configuration. Do not build a custom logging framework.

## Implementation Discipline

Before adding code, stop at the first option that solves the task correctly:

1. Do not build behavior that is not needed.
2. Use the standard library when it already solves the problem.
3. Use an existing repository or platform feature.
4. Use an already-installed dependency.
5. Inline a one-line solution when it remains readable; do not wrap it in a
   one-line helper.
6. Otherwise, write the minimum task-specific code that works.

- No speculative abstractions, dependencies, classes, or boilerplate. Prefer
  deletion over addition and boring code over clever code.
- Question whether a complex requested mechanism is necessary when a simpler
  existing mechanism appears to cover the actual goal. Explain the alternative
  before implementing a materially larger design.
- When two approaches are similarly small, choose the one with correct edge-case
  behavior rather than the more fragile shortcut.
- This economy does not apply to validation at trust boundaries, errors that
  prevent silent data corruption, security, accessibility, scientific
  correctness, or explicitly requested hardware calibration.
- Non-trivial new logic must leave one small runnable check that would fail if
  the behavior breaks. Prefer the smallest focused test or assert-based
  self-check; trivial inline expressions need no dedicated test.
- Prefer the existing Dingo/JPNPE style: clean, readable, and compact.
- Prefer f-strings for string formatting.
- Prefer short, task-specific code over overly general abstractions. It is fine
  to refactor later if a script or approach becomes important.
- Before writing new code, check whether an appropriate function, class, or
  pattern already exists in the repo or in Dingo, and reuse it when possible.
- Strictly avoid unnecessary helper functions, especially one-line helpers or
  short helpers that are only used once. Inline the code unless extracting it
  makes the surrounding logic clearly easier to read.
- In long functions, use brief section-headline comments to mark the main phases
  of the function.
- Avoid adding new classes unless the user asks for one. If a class seems clearly
  necessary, ask first.
- When adding or modifying functions, include a short docstring that explains
  what the function does.
- New scripts and executable modules should start with a short module docstring
  explaining what the file does, especially when they contain a `main()`
  function.

## Testing And Smoke Tests

- For unfinished or actively implemented code, add focused tests for new behavior
  and run them.
- Once a script or task is finished, also run a small smoke test when feasible.
- If the user explicitly says not to run tests or smoke tests for a task, follow
  that instruction for that task.
- Report what was run and whether it passed. If a test or smoke test could not be
  run, say why.

## Working From TODO.md

When the user asks to "work on the TODOs" or similar, use `TODO.md` as the task
queue.

- Start with the highest-priority task whose status is `Ready`.
- Do not work on tasks marked `Needs Review` unless the user explicitly asks.
- For each task you work on, update its status as you go:
  - `Ready` -> `Needs Review` if blocked by a question.
  - `Ready` -> `Needs Review` when implementation and smoke tests are done.
  - `Needs Review` -> `Ready` only after user feedback requests changes.
- Delete accepted tasks from `TODO.md` rather than marking them done.
- Keep working through `Ready` tasks until all are `Needs Review`.
- Follow the repository testing rules: add focused tests for new behavior and run
  small smoke tests in the `tuebingen` environment where feasible.
- Never guess through high-impact scientific or configuration ambiguity. Mark the
  task `Needs Review` and record the question in `TODO.md`.
- When pausing a TODO because of an unresolved question, write the concrete
  question under that task in `TODO.md`, including the relevant file, config, or
  behavior that triggered it.
- When moving a TODO to `Needs Review` after implementation, add a short
  implementation report under that task in `TODO.md`: changed behavior, main
  files touched, tests or smoke tests run, and any known limitations.
