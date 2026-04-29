# Repository Guidelines

## Project Structure & Module Organization
This repository is the `WorldArena` benchmark workspace for dataset analysis. Core Python code lives under `worldarena/`. Keep reusable utilities in `worldarena/common/`, dataset discovery and stats logic in `worldarena/datasets/`, and documentation builders in `worldarena/docs/`. Store runtime configuration in `config/`, helper scripts in `scripts/`, generated analytics in `artifacts/dataset_stats/`, and MkDocs sources in `docs/`. Treat `.cache/` as disposable workspace output, not source content.

When dataset roots or schema assumptions change, update `config/datasets.yaml` and rerun the docs build.

## Coding Style & Naming Conventions
Use Python 3.12+, 4-space indentation, and explicit type hints on public functions. Keep modules focused: scanners should discover data, summarizers should aggregate it, and docs builders should only render outputs. Prefer snake_case for files, functions, and config keys. Use small dataclasses or typed dictionaries when schema grows. Run `ruff check .` before submitting nontrivial changes.

Keep code **concise and efficient**: prefer straightforward algorithms and clear data flow over sprawling conditionals and duplicated logic. Refactor when behavior branches multiply; do not stack unrelated hacks, dead code paths, or copy-paste blocks that become impossible to reason about.

## Version Control (Git)
Use Git for **fine-grained, reviewable history**: work on short-lived branches tied to a task or ticket; keep commits **small and logically scoped** (one concern per commit when practical) so diffs stay easy to audit and bisect. Write commit messages that state *what* changed and *why*, not only branch names or issue ids in the subject.

Avoid mixing unrelated edits in a single commit. Do not commit secrets, large binaries that belong in asset storage, or generated artifacts that the docs build or `clean` command can recreate—unless a task explicitly requires vendoring them with a documented reason.

## Testing Guidelines
Add tests for path parsing, annotation normalization, duration bucketing, and any new registry or config behavior. Mirror package structure under `tests/` where practical. Keep tests deterministic and independent from the large external datasets; use small synthetic fixtures or temporary directories instead of the real data roots whenever possible.

## Commit & Pull Request Guidelines
Follow short, imperative commit messages with optional scopes, e.g. `docs: regenerate dataset analytics` or `datasets: add scanner for game footage`. Pull requests should include:

- the user-facing change summary
- the commands you ran (`pytest`, `mkdocs build --strict`, dataset rebuild command)
- notes on dataset assumptions, exclusions, or schema migrations
- screenshots only when documentation layout or visualizations materially change

## Dataset & Docs Notes
Treat `videos/_review` and backup folders as auxiliary curation assets unless a task explicitly promotes them into benchmark data. Pages under `docs/generated/` are generated output; do not hand-edit them unless you are debugging the generator itself. Keep transient files out of the repo root by using `.cache/` and the `clean` command.
