# Architecture

WorldArena should stay benchmark-first, not pipeline-first. The repo should expose one coherent contract for manifests, model configs, metrics, and reports, while letting local repos, API providers, and heavy metric sources evolve outside the benchmark core.

## Module Boundaries

| Module | Responsibility | What should live there |
| --- | --- | --- |
| `worldarena/benchmark/` | evaluation contract | manifests, protocols, metric adapters, normalization, reporting |
| `worldarena/datasets/` | dataset discovery | raw asset scanning, sidecar discovery, inventory generation |
| `worldarena/models/` | evaluated models and helper runtimes | local-repo adapters, API adapters, provider clients, model config parsing |
| `worldarena/common/` | shared primitives | taxonomy, runtime layout, serialization |
| `config/models/` | unified model registry | one YAML format for local repos and API models |
| `thirdparty/` | linked upstream code | model repos and heavy metric-source checkouts, but not vendored benchmark logic |

The important constraint is that benchmark logic should not fork by model family. A 3D model, a 4D model, an API video model, and a local world model should still enter through the same model config contract and the same report contract.

## Task Families

Current code now carries explicit task metadata through both the model config and manifest layers:

| Field | Examples | Purpose |
| --- | --- | --- |
| `task_families` | `video_generation`, `scene_3d_generation`, `scene_4d_generation`, `world_model`, `reconstruction_3d`, `foundation_oracle` | keeps benchmark expansion organized without creating separate repos |
| `artifact_types` | `image`, `video`, `mesh`, `gaussian_splat`, `point_cloud`, `multiview`, `depth_map` | makes outputs and auxiliary artifacts explicit |
| `control_signals` | `text`, `image`, `video`, `camera_pose`, `actions`, `multi_view`, `depth` | keeps model capability description and metric eligibility aligned |
| `backend_type` | `local_repo`, `python_module`, `sdk_api`, `http_api` | lets API models use the same YAML contract as local models |
| `roles` | `candidate`, `oracle`, `metric_backend`, `reconstruction`, `foundation` | prevents target models and auxiliary evaluator models from being conflated |

For the current formal suites, samples still resolve to `task_family = video_generation`. Physics-style simulator tracks resolve to `task_family = world_model`. This lets the repo grow into 3D/4D/world-model coverage without introducing another benchmark namespace.

## Model Integration Pattern

### Local repos

Use `backend_type: local_repo` and point `repo_root` at a checkout under `thirdparty/` or another local path.

### API models

Use the same YAML file shape, but set:

```yaml
backend_type: http_api
repo_root: null
entrypoint: null
api:
  provider: example
  api_key_env: EXAMPLE_API_KEY
```

The template lives at `config/models/template_video_api.yaml.example`.

This is the intended replacement for copying a large pipeline tree from another repo. If a provider needs custom logic, add a thin adapter in `worldarena/models/adapters/` and a provider client in `worldarena/models/providers/`, but keep the public config format unchanged.

## Metrics And Packs

Metric implementations should stay organized by benchmark dimensions, not by subjective labels like `basic` or `advanced`. The current taxonomy is:

- `long_sequence`
- `action_control`
- `consistency_3d_4d`
- `physics`
- `quality`
- `real_time`

This taxonomy is intentionally coarse-grained. The six keys describe benchmark-level capability coverage, not tiny standalone buckets or a claim that today's public benchmark already spans every possible evaluation mode.

`action_control` is intentionally broader than pure prompt following. It can further split into scene control, camera control, and object/action control, which keeps text-only benchmarks, pose-conditioned runs, and future keyboard/mouse-driven tracks inside one taxonomy.


## Recommended Expansion Order

1. Keep the current formal video benchmark stable.
2. Add API-backed video models through the unified YAML + adapter/provider pattern.
3. Introduce 3D / 4D / reconstruction manifests as new task families, not as separate repos.
4. Add new metric packs, such as real-time calibration or simulator-backed physics, as opt-in dimensions layered onto the same report contract.
5. Only after the contracts are stable, broaden the public model catalog.
