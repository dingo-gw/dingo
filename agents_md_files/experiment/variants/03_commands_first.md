# AGENTS.md

Dingo: gravitational-wave parameter inference via neural posterior estimation (Python, `uv`-managed).

## Commands (use these exactly)
```bash
uv sync                                            # install/sync the environment
uv run pytest                                      # full test suite
uv run pytest tests/gw/test_x.py::test_name        # a single test (fast iteration)
uv run pytest -m "not slow"                        # skip slow markers
uv run black dingo tests                           # format
dingo_pipe <config.ini>                            # run the CLI pipeline
dingo_ls <file.hdf5|model.pt>                      # inspect HDF5 / checkpoint metadata
```

## Where things live
- `dingo/core/` — domain-agnostic ML: posterior models, neural nets, `dataset.py`, `result.py`.
- `dingo/gw/` — GW physics: `likelihood.py`, `domains/`, `waveform_generator/`, `transforms/`, `training/`, `inference/`.
- `dingo/pipe/` — CLI pipeline (bilby_pipe-based). `main.py` builds the completed config + DAG; `parser.py` parses INI; `nodes/` are HTCondor DAG nodes.
- `dingo/asimov/` — Asimov (LVK) integration.

## Conventions
- PEP 604 type hints, NumPy-style docstrings, black formatting.
- Reuse existing helpers; avoid speculative abstractions and one-line helper functions.
- Prefer f-strings.

## Gotchas
- `core/` must not import GW/bilby/LAL — keep the layering clean.
- Frequency domains: `UniformFrequencyDomain` vs `MultibandedFrequencyDomain` — confirm which one applies.
- Config settings are plain nested dicts (not dataclasses); they round-trip through HDF5.

## Don't
- Add dependencies or modify the environment without approval. Don't commit/push.
