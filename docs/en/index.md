# WorldArena

WorldArena is a benchmark for world models. 

Use the documentation in this order if you are getting started:

- **Architecture**: repo boundaries, task families, unified model config, and how local/API models fit the same contract.
- **Benchmark**: evaluation scope, metric surface, reporting contract, and output artifacts.
- **Runtime Setup**: how `cache/`, `thirdparty/`, checkpoints, and adapter repos are expected to be wired together.
- **Dataset Analytics**: collection counts, annotation coverage, mask coverage, caption distributions, hierarchy views, and downloadable summaries.
- **Development**: maintainer workflow, validation commands, and release hygiene.

!!! tip "Recommended first step"
    Start with the benchmark workflow and the model configs under `config/models/`. They define the public contract used by both local adapters and API-backed models.
