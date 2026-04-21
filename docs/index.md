# WorldArena

WorldArena is an extensible benchmark workspace for world model evaluation. This documentation site focuses on one immediate need: turning the current image and video corpora into a continuously regenerable analytics surface that can evolve with the dataset.

Use the pages under **Dataset Analytics** for collection counts, annotation coverage, mask coverage, caption distributions, hierarchy views, and downloadable CSV/JSON summaries.

The repository now also includes a benchmark stack for raw asset inventory, manifest building, suite-aware aggregation, model adapter based generation, and a compact reference-style metric surface for the four formal suites.

!!! tip "Regenerate after data changes"
    Run `worldarena build-docs --config config/datasets.yaml` whenever files are added, removed, or moved in the configured dataset roots.
