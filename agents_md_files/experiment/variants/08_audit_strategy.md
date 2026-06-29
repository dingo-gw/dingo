# DINGO — Codebase Map & Working Notes

## 1. What dingo is
Gravitational-wave parameter inference with neural posterior estimation (~25k LOC, three layers).
Scientifically mature; software-engineering hygiene is lighter in places. Bias toward small,
well-tested changes that respect existing structure.

## 2. Layers
- **`dingo/core/`** — model-agnostic ML/inference (posterior models, nets, datasets, results).
- **`dingo/gw/`** — GW physics (waveforms, likelihood, domains, transforms, training, inference).
- **`dingo/pipe/`** — bilby_pipe-based CLI: generation → sampling → importance sampling → plotting.
- **`dingo/asimov/`** — LVK automation.

## 3. What works well
- Clean core/gw separation; reproducible configs serialized with models; reasonable test foundation; modern packaging (`uv`, `pyproject.toml`).

## 4. Known rough edges (watch for these)
- **Print-style logging** in places instead of a logger — don't add new `print` debugging to committed code.
- **Layering violations** creep in — keep `core/` free of GW/bilby/LAL imports.
- **God-files**: a few large modules (e.g. in `pipe/` and `gw/training/`); read carefully before editing.
- **Frequency-domain assumptions**: `UniformFrequencyDomain` vs `MultibandedFrequencyDomain` mismatches are a recurring bug source.
- **Config plumbing**: settings are nested dicts that must round-trip through HDF5/`.pt`; changing the shape can silently break serialization.

## 5. How to work here
- Reuse existing functions/transforms; prefer the smallest correct change. No speculative abstractions or single-use helpers.
- Add a focused, deterministic test for new logic (RNG seeded).
- Commands: `uv sync`; `uv run pytest [tests/...::test]`; `uv run black dingo tests`.

## 6. Subsystem notes
- **gw**: physics core — waveform generation (LALSuite), likelihood with marginalizations, transform pipelines. Modify training/inference behavior via `transforms/` where possible.
- **core**: keep model-agnostic; this is where flows/embedding nets live.
- **pipe & gw/training**: orchestration and CLI; the completed config is the artifact passed between DAG nodes.

## 7. Caveats
- This is an orientation map, not ground truth — verify against the code before relying on any single claim.

## Boundaries
- Don't add dependencies or modify the environment without approval. Don't commit/push.
