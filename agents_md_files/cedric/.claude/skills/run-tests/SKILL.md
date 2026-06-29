---
name: run-tests
description: Run or select dingo's pytest suite correctly under uv. Use whenever the user asks to run tests, run a specific test, or verify a change in the dingo repo.
---

# Running dingo tests

Tests use **pytest** and must run inside the uv-managed environment. Asimov tests are
excluded by default (`addopts = -m 'not asimov'` in `pyproject.toml`).

```bash
uv run pytest tests/                              # full default suite
uv run pytest tests/gw/path/to/test_x.py         # one file
uv run pytest tests/gw/path/to/test_x.py::test_y # one test
uv run pytest -m "not slow" tests/               # skip slow tests
uv run pytest -m asimov tests/                    # opt-in: asimov integration tests
```

Notes:
- `tests/conftest.py` seeds numpy & bilby RNGs automatically (autouse), so runs are
  deterministic — don't add manual seeding or expect to need it.
- Thread env vars (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) are
  set to `1` via `.claude/settings.json`, matching CI. No action needed; just be aware.
- Test layout mirrors the package: `tests/core/`, `tests/gw/`, `tests/asimov/`.
