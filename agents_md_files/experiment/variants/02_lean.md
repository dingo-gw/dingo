# AGENTS.md

Dingo performs gravitational-wave parameter inference using neural posterior estimation.

## Commands
- Sync deps: `uv sync`
- Run tests: `uv run pytest` (single test: `uv run pytest tests/.../test_x.py::test_name`)
- Format: `uv run black dingo tests`

## Architecture
- `dingo/core/` — model-agnostic ML/inference (posterior models, neural nets, datasets). Must stay domain-agnostic.
- `dingo/gw/` — gravitational-wave specifics: waveforms, likelihood, domains, transforms, training, inference.
- `dingo/pipe/` — command-line pipeline (bilby_pipe-based): data generation → sampling → importance sampling → plotting. `parser.py` parses INI configs; `nodes/` defines HTCondor DAG nodes.
- `dingo/asimov/` — LVK Asimov automation integration.

## Conventions
- Python with type hints; format with black before finishing.
- Reuse existing functions/patterns rather than adding new abstractions.
- Keep `core/` free of GW/bilby/LAL imports.

## Rules
- Don't add dependencies or change the environment without approval.
- Don't commit or push.
