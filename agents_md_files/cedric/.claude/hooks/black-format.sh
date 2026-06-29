#!/usr/bin/env bash
# PostToolUse hook: keep the working tree black-formatted as Claude edits Python files.
# Reads the hook payload on stdin, extracts the edited file path, and formats it in place.
#
# Single source of truth: when a .pre-commit-config.yaml is added to the repo, swap the
# black invocation below for:
#     uv run pre-commit run black --files "$file" >/dev/null 2>&1
# so formatting/lint rules live in one place (the pre-commit config) instead of here.

file=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)

# Only format Python files; no-op for everything else.
if [[ "$file" == *.py && -f "$file" ]]; then
  uv run black -q "$file" >/dev/null 2>&1
fi

exit 0
