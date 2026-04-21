# Repository Guidelines

## Project Structure & Module Organization
This repository is the `WorldArena` benchmark workspace for dataset analysis. Core Python code lives under `worldarena/`. Keep reusable utilities in `worldarena/common/`, dataset discovery and stats logic in `worldarena/datasets/`, and documentation builders in `worldarena/docs/`. Store runtime configuration in `config/`, helper scripts in `scripts/`, generated analytics in `artifacts/dataset_stats/`, and MkDocs sources in `docs/`. Treat `.cache/` as disposable workspace output, not source content.

## Build, Test, and Development Commands
Use `uv` for all local development:

- `uv sync` creates or updates the managed environment.
- `uv run worldarena build-docs --config config/datasets.yaml` scans datasets, refreshes CSV/JSON artifacts, and regenerates `docs/generated/`.
- `uv run mkdocs serve` previews the documentation site locally.
- `uv run mkdocs build --strict` validates the site build.
- `uv run pytest` runs unit tests.
- `uv run ruff check .` runs lint checks.
- `uv run worldarena clean` removes transient caches and local build output.

When dataset roots or schema assumptions change, update `config/datasets.yaml` and rerun the docs build.

## Coding Style & Naming Conventions
Use Python 3.12+, 4-space indentation, and explicit type hints on public functions. Keep modules focused: scanners should discover data, summarizers should aggregate it, and docs builders should only render outputs. Prefer snake_case for files, functions, and config keys. Use small dataclasses or typed dictionaries when schema grows. Run `ruff check .` before submitting nontrivial changes.

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
