# Architecture

WorldAtlas Arena should stay benchmark-first, not pipeline-first. The repo should expose one coherent contract for manifests, model configs, metrics, and reports, while letting local repos, API providers, and heavy metric sources evolve outside the benchmark core.

## Module Boundaries

| Module | Responsibility | What should live there |
| --- | --- | --- |
| `worldarena/benchmark/` | evaluation contract | manifests, protocols, metric adapters, normalization, reporting |
| `worldarena/datasets/` | dataset discovery | raw asset scanning, sidecar discovery, inventory generation |
| `worldarena/models/` | evaluated models and helper runtimes | local-repo adapters, API adapters, provider clients, model config parsing |
| `worldarena/common/` | shared primitives | taxonomy, runtime layout, serialization |
| `config/models/` | unified model registry | one YAML format for local repos and API models |
| `thirdparty/` | upstream code | model repositories and pinned metric-source checkouts; metric adapters stay in the main package |

Official metric orchestration lives in `worldarena/benchmark/official_backends/`.
Those adapters reuse upstream checkouts from `thirdparty/`; upstream source is
not copied into the Python package.

The important constraint is that benchmark logic should not fork by model family. A 3D model, a 4D model, an API video model, and a local world model should still enter through the same model config contract and the same report contract.

## Core Contracts

WorldAtlas Arena's public surface is data-first. Internal runners and adapters can evolve, but the artifacts below are the contracts that make runs comparable and debuggable:

| Contract | Owner | Stable fields | Extension rule |
| --- | --- | --- | --- |
| Manifest sample | `worldarena_manifest.jsonl`, backed by `BenchmarkSample` | `sample_id`, `suite`, `split`, `reference_path`, `prediction_stem`, conditioning fields, prompts, domain tags, task metadata, supervision sidecars, and media probe fields | Add nullable fields for new supervision or artifact metadata. Do not infer benchmark semantics from directory names once the manifest has been built. |
| Model runtime config | `config/models/*.yaml`, backed by `ModelRuntimeConfig` | model identity, `backend_type`, runtime entrypoint or API provider, `supported_suites`, `task_families`, `artifact_types`, `control_signals`, `roles`, `generation`, and `metric_overrides` | Keep one YAML shape for local repos, Python modules, SDK APIs, and HTTP APIs. Add provider-specific details under `api` or `metadata` rather than changing the top-level contract. |
| Prediction artifact | adapter/provider outputs under `predictions/` | `<suite>/<prediction_stem>.<ext>` or `<prediction_stem>.<ext>`, with extensions chosen from the prediction modality | The prediction is the evaluated artifact. Extra frames, meshes, depth maps, or logs should be explicit sidecars and should not change the required prediction filename. |
| Metric output | `MetricOutput` inside `per_sample_scores.jsonl` | `raw`, `normalized`, `backend`, `details`, `protocol_name`, `eligibility_status`, and `error` | Store the raw measurement and the normalized score separately. Use `official`, `diagnostic`, and `not_applicable` statuses instead of encoding aggregation behavior in metric names. |
| Report bundle | `run_manifest.json`, summaries, breakdowns, and `leaderboard.csv` | run metadata, missing prediction count, reported metrics, official summaries, non-official summaries, failure summaries, and domain breakdowns | Preserve official, diagnostic, not-applicable, and failure channels as separate outputs. Do not introduce a benchmark-wide total score unless the weighting contract is explicitly versioned. |

The lifecycle is intentionally one-directional: dataset scanners create inventory, the manifest freezes sample semantics, model adapters create predictions keyed by `prediction_stem`, metrics read manifest plus prediction artifacts, and reports aggregate only from metric outputs. Adapters should not inspect dataset layout to decide eligibility, and metrics should not branch on model family.

When extending the benchmark, prefer existing axes before creating new surfaces: use `task_family` for new task classes, `artifact_type` for new outputs, `control_signals` for new inputs, `roles` for auxiliary models, and metric dimensions for new scoring groups. If a new enum value is needed, update the taxonomy, config defaults, tests, and docs together.

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
