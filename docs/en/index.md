# WorldAtlas Arena

WorldAtlas Arena is developed by [OpenEnvision](https://openenvision.github.io/index.html), a joint research hub for open vision intelligence and world modeling.

## Summary

WorldAtlas Arena evaluates generative world models along a graded capability hierarchy—from perception and persistent scene modeling through manipulation, dynamics, and closed-loop interaction—rather than by surface fidelity alone.

## Capability Gradient

World models are often judged by the visual plausibility of their outputs, a criterion that can obscure sharp differences in underlying capability. A more informative view arranges these capabilities along a graded hierarchy:

- **Perception** — recovering information already present in an observation
- **Modeling** — maintaining a persistent internal state beyond any single frame
- **Manipulation** — transforming that state under an explicit constraint
- **Dynamics** — propagating state through time under physical and causal law
- **Interaction** — producing action-conditioned, counterfactually valid rollouts

Competence at one level does not entail competence at the levels above it.

WorldAtlas Arena evaluates models against this gradient rather than against surface fidelity alone: its suites and metrics are designed to identify which level of capability a given system actually exhibits, from perception and scene modeling through physically consistent dynamics and closed-loop interaction. The repository mirrors this separation of concerns, with evaluation logic, adapter contracts, third-party checkouts, and cached checkpoints organized to evolve independently.

## Documentation Map

For the formal benchmark framing and task decomposition, start with the [Benchmark Overview](benchmark/index.md). The rest of the documentation set is organized as follows:

<div class="doc-map">
<div class="doc-map-grid">
<a class="doc-map-card doc-map-card--benchmark" href="benchmark/">
  <span class="doc-map-card-kicker">01 · Evaluation</span>
  <span class="doc-map-card-title">Benchmark</span>
  <span class="doc-map-card-stat">4 suites</span>
  <span class="doc-map-card-desc">Evaluation scope, suites, dimensions, and reporting contract.</span>
</a>
<a class="doc-map-card doc-map-card--metrics" href="benchmark/metrics/">
  <span class="doc-map-card-kicker">02 · Metrics</span>
  <span class="doc-map-card-title">Metrics</span>
  <span class="doc-map-card-stat">39 docs</span>
  <span class="doc-map-card-desc">The full metric surface grouped by evaluation dimension.</span>
</a>
<a class="doc-map-card doc-map-card--runtime" href="thirdparty/">
  <span class="doc-map-card-kicker">03 · Runtime</span>
  <span class="doc-map-card-title">Runtime Setup</span>
  <span class="doc-map-card-stat">Adapters</span>
  <span class="doc-map-card-desc">Third-party checkouts, checkpoints, and adapter wiring.</span>
</a>
<a class="doc-map-card doc-map-card--dataset" href="overview/">
  <span class="doc-map-card-kicker">04 · Corpus</span>
  <span class="doc-map-card-title">Dataset Analytics</span>
  <span class="doc-map-card-stat">11,653 items</span>
  <span class="doc-map-card-desc">Collection counts, annotation coverage, and media statistics.</span>
</a>
</div>
</div>

!!! tip "Recommended first step"
    Start with the [benchmark overview](benchmark/index.md) and the model configs under `config/models/`. They define the public contract used by both local adapters and API-backed models.
