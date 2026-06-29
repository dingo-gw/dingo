# cedric — full Claude Code agent config (the user's "11th example")

This run does **not** use one of the authored variant files. It uses the complete `cedric/`
configuration as-is:
- `cedric/AGENTS.md` (~123 lines) — tool-agnostic project spec (setup, commands, architecture, conventions, testing rules, end-to-end smoke test, gotchas, do/don't).
- `cedric/CLAUDE.md` (~18 lines) — thin Claude-Code layer that `@AGENTS.md`-imports the spec and points at the automation below.
- `cedric/.claude/`:
  - `settings.json` — thread-pinned env (`OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`), an allow-list limited to `uv run pytest/black/make`, `uv sync`, `dingo_ls`, and read-only git (no commit/push), plus a `PostToolUse` hook.
  - `hooks/black-format.sh` — auto-formats every edited `.py` with `uv run black`.
  - `agents/dingo-reviewer.md` — read-only domain reviewer (layering, conventions, config, domains, tests, hygiene).
  - `commands/format.md` — `/format` (black on changed or whole tree).
  - `skills/{run-tests,smoke-test,build-docs}/SKILL.md` — correct-invocation shortcuts.

**Caveat (recorded in the report):** a spawned sub-agent does not auto-fire Claude Code hooks/skills,
so for this benchmark the cedric config's *content* (AGENTS.md + CLAUDE.md + the skill/subagent/hook
text and the "format every edited .py with black" behavior) is injected into the agent's prompt.
It therefore measures cedric's guidance, not its live hook automation.
