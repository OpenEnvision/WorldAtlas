# WorldArena

WorldArena is a benchmark and dataset analytics workspace for world models. The repository is organized so that benchmark logic, adapter contracts, third-party repos, and cached checkpoints can evolve independently.

Use the documentation in this order if you are getting started:

- **Benchmark**: evaluation scope, metric surface, reporting contract, and output artifacts.
- **Environment**: install profiles, `uv` commands, self-check workflow, and runtime layering.
- **Runtime Setup**: how `cache/`, `third_party/`, checkpoints, and adapter repos are expected to be wired together.
- **Dataset Analytics**: collection counts, annotation coverage, mask coverage, caption distributions, hierarchy views, and downloadable summaries.
- **Development**: maintainer workflow, validation commands, and release hygiene.

!!! tip "Recommended first step"
    Run `worldarena prepare-layout`, then `worldarena check-env`, before integrating upstream model repos or large checkpoint caches. This keeps the checkout publishable while still making dependency gaps obvious early.
