# Cache Layout

This directory is the local home for reusable model and evaluation assets.

Expected content:

- `hfd/`: a Hugging Face style cache root used by reference metrics and bundled model configs.

Recommended setup:

```bash
uv run worldarena prepare-layout
```

If you already have a shared cache on the machine, reuse it instead of copying weights:

```bash
uv run worldarena prepare-layout --shared-hf-cache /path/to/shared/hf-cache
```

Validate that the cache and metric-source layout resolve correctly:

```bash
uv run worldarena check-env
```

This directory is intentionally git-ignored except for this README. Do not commit downloaded checkpoints, symlinks, or large generated assets.
