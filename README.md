# WorldArena

WorldArena is an extensible benchmark workspace for world model evaluation, dataset inspection, and documentation generation. The repository is designed to stay stable at the benchmark-contract layer while model adapters, third-party repos, checkpoints, and metric backends evolve independently.

This repo does not vendor upstream model repositories or redistribute model weights. Instead, it provides:

- the benchmark inventory, manifest, and reporting contract
- official metric execution for the formal suites
- a unified model config contract for local repos and API models under `config/models/`
- a documented runtime layout for `cache/` and `third_party/`
- a runtime self-check command for open-source users

## Repo Shape

WorldArena should stay benchmark-first, not pipeline-first:

- `worldarena/benchmark/`: manifests, metrics, normalization, evaluation runner, reporting
- `worldarena/models/`: evaluated model adapters plus optional API-provider clients
- `worldarena/datasets/`: dataset discovery and inventory logic
- `worldarena/common/`: shared taxonomy, runtime layout, environment inspection, serialization
- `config/models/`: one YAML schema for local-repo models and API-backed models

The current code now carries explicit `backend_type`, `task_families`, `artifact_types`, `control_signals`, and `roles` metadata. That is the mechanism for scaling from today's video benchmark into 3D, 4D, reconstruction, and world-model coverage without creating a second framework inside the repo.

## Install Profiles

Core benchmark and dataset analytics:

```bash
uv sync --python 3.10
```

Official metric backends:

```bash
uv sync --python 3.10 --extra eval --extra quality
```

Temporal calibration metrics:

```bash
uv sync --python 3.10 --extra temporal-calibration
```

Heavy video metrics:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
./.venv/bin/python -m spacy download en_core_web_sm
```

Heavy video metrics also need a compatible metric-source checkout wired either through `WORLDARENA_METRIC_SOURCE_ROOT` or under `third_party/`.

`video-metrics` includes the heavier official dependencies such as `cvxpy`, `iopath`, `pycocotools`, `causal-conv1d`, and `mamba-ssm`. If `mamba-ssm` wheel resolution fails on your machine, install the exact `causal-conv1d` / `mamba-ssm` pair recommended by the compatible metric-source checkout and rerun `worldarena check-env`.

Temporal calibration metrics also expect a `Pulse-of-Motion` checkout, either under `third_party/Pulse-of-Motion` or wired through `WORLDARENA_PULSE_OF_MOTION_ROOT`. The upstream checkpoint does not need to be committed locally in advance; it can be auto-downloaded on first use when `huggingface_hub` is available.

Integrated world-model adapters:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

## Prepare Runtime Layout

Create the open-source-friendly local layout:

```bash
uv run worldarena prepare-layout
```

Reuse an existing checkpoint cache and link local model repos into the expected locations:

```bash
uv run worldarena prepare-layout \
  --shared-hf-cache /path/to/shared/hf-cache \
  --repo-link lingbot-world=/path/to/lingbot-world \
  --repo-link Infinite-World=/path/to/Infinite-World \
  --repo-link Lyra-1=/path/to/Lyra-1 \
  --repo-link Lyra-2=/path/to/Lyra-2 \
  --repo-link WorldCam=/path/to/WorldCam \
  --repo-link HunyuanWorld-1.0=/path/to/HunyuanWorld-1.0 \
  --repo-link worldfm=/path/to/worldfm \
  --repo-link Matrix-Game-1=/path/to/Matrix-Game-1 \
  --repo-link Matrix-Game-2=/path/to/Matrix-Game-2 \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0 \
  --repo-link metric-source=/path/to/compatible-metric-checkout \
  --repo-link Pulse-of-Motion=/path/to/Pulse-of-Motion
```

Default contract:

```text
cache/
  hfd/
third_party/
  lingbot-world/
  Infinite-World/
  Lyra-1/
  Lyra-2/
  WorldCam/
  HunyuanWorld-1.0/
  worldfm/
  Matrix-Game-1/
  Matrix-Game-2/
  Matrix-Game-3.0/
  metric-source/
  Pulse-of-Motion/
```

Tracked guidance lives in `cache/README.md` and `third_party/README.md`. Both directories are git-ignored except for those READMEs so that local clones, symlinks, and large checkpoints stay out of version control.

Inspect the local setup after wiring it:

```bash
uv run worldarena check-env
```

`check-env` is a static readiness probe. It reports layout, dependency, metric-source, and GPU status without instantiating heavy metric classes.

## Benchmark Workflow

WorldArena keeps four formal suites as the public benchmark core:

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

Official headline reporting is a per-suite metric vector, not a suite-level scalar and not a benchmark-wide total score.

Build inventory and manifest:

```bash
uv run worldarena build-inventory --config config/benchmark.yaml
uv run worldarena build-manifest --config config/benchmark.yaml
```

Generate outputs with an adapter config:

```bash
uv run worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Evaluate an existing predictions folder:

```bash
uv run worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root /path/to/predictions
```

Or run the one-shot path:

```bash
uv run worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Expected prediction layout:

```text
predictions/
  image_static/<sample_id>.png
  image_dynamic/<sample_id>.png
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

Run artifacts include:

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `diagnostic_metric_summary.json` when any metric is non-official for eligibility reasons
- `domain_breakdown.json`
- `leaderboard.csv`

`motion_accuracy` is the heaviest dynamic metric. The default benchmark config enables SAM2 video/state offload so that the official path can run on 80 GB cards more reliably. On multi-GPU hosts, `benchmark.metric_runtime.motion_accuracy` also supports device splitting such as `sam2_device: cuda:0` and `flow_device: cuda:1`.

## Official Reporting Contract

Current official metric dimensions are:

- `content_alignment`: `prompt_alignment`
- `style_consistency`: `style_consistency`
- `perceptual_quality`: `perceptual_quality`
- `camera_control`: `camera_error`
- `object_control`: `object_detection`
- `geometry_consistency`: `reprojection_error`
- `photometric_consistency`: `optical_flow_aepe`
- `motion_control`: `motion_accuracy`
- `motion_dynamics`: `motion_magnitude`, `motion_smoothness`
- `temporal_calibration`: `physical_fps_error`, `physical_fps_pct_error`, `physical_fps_intra_video_cv`

Current suite assignments are:

- `image_static`: `style_consistency`, `perceptual_quality`
- `image_dynamic`: `style_consistency`, `perceptual_quality`
- `video_static`: `style_consistency`, `perceptual_quality`, `camera_error`, `object_detection`, `reprojection_error`, `optical_flow_aepe`
- `video_dynamic`: `style_consistency`, `perceptual_quality`, `motion_accuracy`, `motion_magnitude`, `motion_smoothness`, `physical_fps_error`, `physical_fps_pct_error`, `physical_fps_intra_video_cv`

`prompt_alignment` is still computed, but on self-conditioned samples with template-derived prompts it is reported as `weak_supervision` and routed to diagnostics instead of the official aggregate.

`object_detection` is also routed to `weak_supervision` when object prompts have to be derived from caption text instead of explicit object annotations.

Non-formal assets stay outside official headline reporting and are routed into separate `auxiliary` or `experimental` reporting.

## Adapter Runtime Contract

Bundled model configs now use project-relative defaults so the repository can be published without private absolute paths:

- `config/models/lingbot_world.yaml`
  - `repo_root: third_party/lingbot-world`
  - `checkpoint_dir: cache/hfd/robbyant--lingbot-world-base-cam`
- `config/models/infinite_world.yaml`
  - `repo_root: third_party/Infinite-World`
  - `checkpoint_dir: cache/hfd/MeiGen-AI--Infinite-World`
- `config/models/lyra1.yaml`
  - `repo_root: third_party/Lyra-1`
  - `checkpoint_dir: cache/hfd/Lyra`
- `config/models/lyra2.yaml`
  - `repo_root: third_party/Lyra-2`
  - `checkpoint_dir: cache/hfd/Lyra-2.0`
- `config/models/worldcam.yaml`
  - `repo_root: third_party/WorldCam`
  - `checkpoint_dir: cache/hfd/worldcam--worldcam`
- `config/models/hunyuanworld.yaml`
  - `repo_root: third_party/HunyuanWorld-1.0`
  - `checkpoint_dir: cache/hfd/tencent--HunyuanWorld-1`
- `config/models/worldfm.yaml`
  - `repo_root: third_party/worldfm`
  - `checkpoint_dir: cache/hfd/inspatio--worldfm`
- `config/models/matrix_game_1.yaml`
  - `repo_root: third_party/Matrix-Game-1`
  - `checkpoint_dir: cache/hfd/Matrix-Game-1.0`
- `config/models/matrix_game_2.yaml`
  - `repo_root: third_party/Matrix-Game-2`
  - `checkpoint_dir: cache/hfd/Skywork--Matrix-Game-2.0`
- `config/models/matrix_game.yaml`
  - `repo_root: third_party/Matrix-Game-3.0`
  - `checkpoint_dir: cache/hfd/Matrix-Game-3.0`

Bundled configs also use a single schema for model capability metadata:

- `backend_type`: `local_repo`, `python_module`, `sdk_api`, or `http_api`
- `task_families`: for example `video_generation` or `world_model`
- `artifact_types`: for example `video`, `image`, `mesh`, `point_cloud`
- `control_signals`: for example `text`, `image`, `video`, `camera_pose`, `actions`
- `roles`: `candidate`, `oracle`, `metric_backend`, `reconstruction`, or `foundation`

API-backed models should use the same YAML shape and set `repo_root: null`. The starter template lives at `config/models/template_video_api.yaml.example`.

The current API path keeps one benchmark-facing contract for both local repos and remote generation services:

- `backend_type: http_api`
- the provider contract lives under `model.api`
- the generation knobs stay under `model.generation`
- the same `worldarena generate`, `evaluate`, and `run-model` commands work unchanged

`config/models/kling_api.yaml.example` is the concrete example for this path.

If the upstream model code needs a separate CUDA-enabled interpreter, the configs still support `python_bin: auto`, which prefers `.runtime312/bin/python` when present.

`Infinite-World`, `Lyra-1`, `Lyra-2`, `WorldFM`, and `HunyuanWorld` are the main exceptions today: the bundled configs pin `python_bin: .runtime310/bin/python` because these upstream stacks typically want Python 3.10 plus their own CUDA-compatible environments.

Example runtime bootstrap for adapters that need a 3.12 side environment:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  modelscope sentencepiece protobuf pynvml opencv-python==4.10.0.84
```

Optional model-side environment overrides live under `model.env` in the YAML configs. Use that block for `CUDA_VISIBLE_DEVICES`, `HF_HOME`, or other adapter-specific runtime settings.

For `lingbot-world`, the `model.generation` block is intended to stay close to the upstream CLI. Beyond the core fields (`task`, `size`, `frame_num`, `sample_solver`), the adapter also forwards controls such as `sample_steps`, `sample_shift`, `sample_guide_scale`, `ulysses_size`, `offload_model`, `t5_fsdp`, `t5_cpu`, `dit_fsdp`, `use_prompt_extend`, `prompt_extend_*`, and `action_string_by_suite`.

For `Infinite-World`, the `model.generation` block keeps the benchmark-side knobs small: `config_path`, `bucket_config_name`, `num_chunks`, `num_sampling_steps`, `sample_shift`, `sample_guide_scale`, `fps`, and an `action_plan` that expands suite-specific symbolic interactions like `forward`, `forward_right`, and `camera_left` into the framewise move/view JSON expected by the upstream model.

For `HunyuanWorld`, the adapter targets static suites only. It uses the reference image or first video frame to run `demo_panogen.py`, feeds the resulting `panorama.png` into `demo_scenegen.py`, and archives the generated `mesh_layer*.ply` files and intermediates into a single `.zip`. This is generation-only plumbing for now: the current `evaluate` path does not score those mesh archives.

For `WorldFM`, the adapter converts each sample's `poses.npy` and `intrinsics.npy` into the upstream `meta.json` format and then runs `run_pipeline.py` against the cached first frame. The bundled config intentionally stays narrow: `model_filename`, `vae_subdir`, `step`, `image_size`, `cfg_scale`, `render_size`, `fps`, and `gpu_index`. It is registered for `video_static` only because the upstream model is a single-image, pose-driven novel-view generator rather than a dynamic scene model.

## Dataset Analytics And Docs

Refresh the dataset analytics pages:

```bash
uv run worldarena build-docs --config config/datasets.yaml
```

Preview the documentation site locally:

```bash
uv run mkdocs serve
```

Additional documentation:

- `docs/architecture.md`: repo boundaries, task families, unified model config, and API integration pattern
- `docs/benchmark/`: benchmark scope, metric policy, reporting contract, and per-metric notes
- `docs/environment.md`: install profiles, self-check flow, runtime layering
- `docs/third_party.md`: third-party repos, cache layout, and checkpoint integration
- `docs/development.md`: maintainer workflow, validation commands, release hygiene
