---
description: Format dingo Python code with black (whole tree or just changed files)
---

Format the codebase with black using the project environment.

- If there are uncommitted changes, format only the changed Python files:
  `git diff --name-only --diff-filter=ACM -- '*.py' | xargs -r uv run black`
  and also include staged files: `git diff --cached --name-only --diff-filter=ACM -- '*.py'`.
- Otherwise (or if the user asks for the whole tree), run: `uv run black dingo tests`.

After running, report which files black reformatted (or "all files already formatted").
Do not commit; just format the working tree.
