# CLAUDE.md

Claude Code entry point for the dingo repo. The shared, tool-agnostic project knowledge
lives in **AGENTS.md** (imported below) — keep edits to project facts there, not here, so
every agent stays in sync. This file holds only Claude-Code-specific pointers.

## Claude Code extras in this repo (`.claude/`)

- **Skills** (auto-triggered): `run-tests`, `smoke-test`, `build-docs`.
- **Subagent**: `dingo-reviewer` — domain-aware read-only code review (`@dingo-reviewer`).
- **Slash command**: `/format` — black the working tree or just changed files.
- **Hook**: `black-format.sh` runs black on every `.py` file after you edit it.
- **Settings** (`.claude/settings.json`): pins thread env vars to match CI and allowlists
  safe commands. Personal overrides go in an untracked `.claude/settings.local.json`.

## Shared project guidance

@AGENTS.md
