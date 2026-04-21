# Architecture

WorldArena should stay benchmark-first, not pipeline-first. The repo should expose one coherent contract for manifests, model configs, metrics, and reports, while letting local repos, API providers, and heavy metric sources evolve outside the benchmark core.

## Module Boundaries

| Module | Responsibility | What should live there |
| --- | --- | --- |
| `worldarena/benchmark/` | evaluation contract | manifests, protocols, metric adapters, normalization, reporting |
| `worldarena/datasets/` | dataset discovery | raw asset scanning, sidecar discovery, inventory generation |
| `worldarena/models/` | evaluated models and helper runtimes | local-repo adapters, API adapters, provider clients, model config parsing |
| `worldarena/common/` | shared primitives | taxonomy, runtime layout, environment inspection, serialization |
| `config/models/` | unified model registry | one YAML format for local repos and API models |
| `third_party/` | linked upstream code | model repos and heavy metric-source checkouts, but not vendored benchmark logic |

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

Use `backend_type: local_repo` and point `repo_root` at a linked checkout under `third_party/` or a local path via `worldarena prepare-layout --repo-link`.

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

Metric implementations should stay organized by evaluation dimension, not by subjective labels like `basic` or `advanced`. The current repo already follows this for the integrated video core:

- `content_alignment`
- `style_consistency`
- `perceptual_quality`
- `camera_control`
- `object_control`
- `geometry_consistency`
- `photometric_consistency`
- `motion_control`
- `motion_dynamics`

Additional metric sources should attach as dimension-specific packs rather than becoming a second benchmark system. For example:

- `Pulse-of-Motion` should live under a `temporal_calibration` dimension.
- 3D reconstruction quality should live under geometry/reconstruction dimensions.
- 4D world consistency should live under temporal + geometric dimensions.

That keeps one report surface while still allowing richer evaluation than the current video core.

## Recommended Expansion Order

1. Keep the current formal video benchmark stable.
2. Add API-backed video models through the unified YAML + adapter/provider pattern.
3. Introduce 3D / 4D / reconstruction manifests as new task families, not as separate repos.
4. Add new metric packs, such as temporal calibration, as opt-in dimensions layered onto the same report contract.
5. Only after the contracts are stable, broaden the public model catalog.
