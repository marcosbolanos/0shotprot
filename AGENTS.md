# AGENTS

## Environment and command execution

- This repository uses `uv` for Python environment and dependency management.
- Run Python and project commands with `uv run`.
- Prefer `uv run <command>` over invoking `python` or tools directly.

Examples:

- `uv run python -m pytest`
- `uv run python src/prospero/runners/plot_benchmarks.py --help`
