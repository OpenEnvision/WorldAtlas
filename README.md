# WorldArena

WorldArena is an extensible benchmark workspace for world model evaluation, dataset inspection, and documentation generation. The repository is designed to stay stable at the benchmark-contract layer while model adapters, third-party repos, checkpoints, and metric backends evolve independently.

This repo does not vendor upstream model repositories or redistribute model weights. Instead, it provides:

- the benchmark inventory, manifest, and reporting contract
- official metric execution for the formal suites
- a unified model config contract for local repos and API models under `config/models/`
- a documented runtime layout for `ckpt/` and `thirdparty/`

All commands below assume the project's conda environment is already activated.

## Repo Shape

WorldArena should stay benchmark-first, not pipeline-first:

- `worldarena/benchmark/`: manifests, metrics, normalization, evaluation runner, reporting
- `worldarena/models/`: evaluated model adapters plus optional API-provider clients
- `worldarena/datasets/`: dataset discovery and inventory logic
- `worldarena/common/`: shared taxonomy, runtime layout, serialization
- `config/models/`: one YAML schema for local-repo models and API-backed models

The current code now carries explicit `backend_type`, `task_families`, `artifact_types`, `control_signals`, and `roles` metadata. That is the mechanism for scaling from today's video benchmark into 3D, 4D, reconstruction, and world-model coverage without creating a second framework inside the repo.

## Benchmark Workflow

WorldArena keeps four formal suites as the public benchmark core:

- `image_static`
- `image_dynamic`
- `video_static`
- `video_dynamic`

Official headline reporting is a per-suite metric vector, not a suite-level scalar and not a benchmark-wide total score.

Build inventory and manifest:

```bash
worldarena build-inventory --config config/benchmark.yaml
worldarena build-manifest --config config/benchmark.yaml
```

Generate outputs with an adapter config:

```bash
worldarena generate \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Evaluate an existing predictions folder:

```bash
worldarena evaluate \
  --config config/benchmark.yaml \
  --predictions-root /path/to/predictions
```

Or run the one-shot path:

```bash
worldarena run-model \
  --config config/benchmark.yaml \
  --model-config config/models/lingbot_world.yaml
```

Expected prediction layout:

```text
predictions/
  image_static/<sample_id>.mp4
  image_dynamic/<sample_id>.mp4
  video_static/<sample_id>.mp4
  video_dynamic/<sample_id>.mp4
```

Run artifacts include:

- `run_manifest.json`
- `per_sample_scores.jsonl`
- `per_metric_summary.json`
- `diagnostic_metric_summary.json` when any diagnostic metric returns a numeric score
- `not_applicable_summary.json` when any metric is skipped for missing strong supervision
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
- `physics_fidelity`: `pmf`, `fvmd`
- `temporal_calibration`: `physical_fps_error`, `physical_fps_pct_error`, `physical_fps_intra_video_cv`

Current suite assignments are:

- `image_static`: image-conditioned video rollout. It now runs the video-output metric superset where inputs permit: prompt/object/style/quality, prediction-only geometry and motion metrics, temporal calibration, depth-collision, self-consistency, and long-sequence diagnostics. Metrics requiring reference video, pose, or masks return unavailable when those sidecars are absent.
- `image_dynamic`: same video-output metric superset as `image_static`, with dynamic prompts and masks used when available.
- `video_static`: `style_consistency`, `perceptual_quality`, `camera_error`, `object_detection`, `reprojection_error`, `optical_flow_aepe`, `fvmd`
- `video_dynamic`: `style_consistency`, `perceptual_quality`, `object_permanence`, `pmf`, `fvmd`, `motion_accuracy`, `motion_magnitude`, `motion_smoothness`, `physical_fps_error`, `physical_fps_pct_error`, `physical_fps_intra_video_cv`

The `image_static`, `image_dynamic`, and `video_dynamic` suites also emit diagnostic-only long-video metrics via the bundled native `Long-Sequence` backend: `long_sequence_subject_consistency`, `long_sequence_background_consistency`, `long_sequence_motion_smoothness`, `long_sequence_dynamic_degree`, `long_sequence_aesthetic_quality`, and `long_sequence_imaging_quality`.

All image and video formal suites also emit diagnostic `depth_collision`, which uses Video-Depth-Anything metric depth over a center crop to flag camera-path collisions. Its default checkpoint path is `ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth`.

All image and video formal suites also emit diagnostic `reconstruction_consistency`, the World-R1 `S_recon` signal. It reconstructs a Depth-Anything-3 Gaussian representation from the prediction, renders it back along the input path, and reports `clip(1 - mean LPIPS, 0, 1)` as a prediction-only consistency score.

`prompt_alignment` is computed only when the sample has explicit instruction supervision and is not self-conditioned. It always evaluates video predictions: image suites are image-conditioned video rollouts, so their `prompt_target` must describe the intended video, not a still-image output. Otherwise the metric is reported as `not_applicable` and excluded from both official and diagnostic aggregates.

`fvmd` is a suite-level Fréchet Video Motion Distance metric, so it is computed once per video suite from the set of valid prediction/reference video pairs and written to `distribution_metric_details.json` alongside the normal summary files. Its default source checkout is `thirdparty/FVMD-frechet-video-motion-distance`.

`semantic_alignment` is also emitted for image suites as a diagnostic caption-grounded signal. It remains outside the official aggregate.

`object_detection` is computed only from explicit structured object annotations. Caption-derived object prompts are reported as `not_applicable`, not scored.

Non-formal assets stay outside official headline reporting and are routed into separate `auxiliary` or `experimental` reporting.

## Checkpoint Contract

`WorldArena/ckpt` is the default checkpoint root. Large-disk deployments may point the same layout elsewhere with `WORLDARENA_CHECKPOINT_ROOT=/path/to/ckpt`. All model checkpoints, benchmark weights, reference metric weights, and HF/Torch runtime caches resolve under the active checkpoint root. `thirdparty/` keeps source checkouts only, and `cache/` is scratch runtime output only.

Removed path mechanisms:

- `cache/hfd`
- `~/.cache/worldarena/**`
- `~/.cache/vbench/**`
- `thirdparty/**/ckpts`
- adapter/backend-specific fallback search chains

The tested open-source evaluation environment is `conda` env `worldarena`, Python 3.10.14, PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1, and ffmpeg exposed on PATH. See `docs/en/thirdparty.md` or `docs/zh/thirdparty.md` for the exact setup order, metric checkpoint layout, and `decord` installation note.

Simulator-backed physics evaluation uses the same WorldArena report contract but depends on a separate external `physics-pipeline` checkout and its own Python runtime. WorldArena invokes it through `WORLDARENA_PHYSICS_PYTHON_BIN` or the per-case `python_bin` metadata. The in-tree validation command is `python -m pytest tests/test_physics_simulator_derived_metrics.py`; a full simulator run additionally requires `benchmark_export.py`, `main_part2.py`, `benchmark_eval.py`, and pipeline assets under the configured `--physics-pipeline-root`.

Bundled model defaults:

| Config | Source checkout | Checkpoint path(s) under `ckpt/` |
| --- | --- | --- |
| `lingbot_world.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam` |
| `lingbot_world_base_act_preview.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-act-preview`, `generation.shared_checkpoint_dir=lingbot-world-base-cam` |
| `lingbot_world_fast.yaml` | `thirdparty/lingbot-world` | `lingbot-world-base-cam`, `generation.fast_checkpoint_dir=lingbot-world-fast` |
| `inspatio_world.yaml` | `thirdparty/inspatio-world` | `inspatio--world` |
| `infinite_world.yaml` | `thirdparty/Infinite-World` | `MeiGen-AI--Infinite-World` |
| `neoverse.yaml` | `thirdparty/NeoVerse` | `Yuppie1204--NeoVerse` |
| `lyra1.yaml` | `thirdparty/Lyra-1` | `Lyra` plus companion `nvidia--GEN3C-Cosmos-7B`, `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`, `google-t5--t5-11b`, `Ruicheng--moge-vitl` |
| `lyra2.yaml` | `thirdparty/Lyra-2` | `Lyra-2.0` plus companion `depth-anything--DA3NESTED-GIANT-LARGE`, `google/umt5-xxl`, `Ruicheng--moge-vitl` |
| `gen3c.yaml` | `thirdparty/Lyra-1` | `nvidia--GEN3C-Cosmos-7B` plus companion `nvidia--Cosmos-Tokenize1-CV8x8x8-720p`, `google-t5--t5-11b`, `Ruicheng--moge-vitl` |
| `flashworld.yaml`, `flashworld_image_static.yaml` | `thirdparty/FlashWorld` | `imlixinyang--FlashWorld` |
| `worldcam.yaml` | `thirdparty/WorldCam` | `worldcam--worldcam` plus companion `Wan-AI--Wan2.1-T2V-1.3B` |
| `hunyuanworld.yaml` | `thirdparty/HunyuanWorld-1.0` | `HunyuanWorld-1` plus companion `ZIM/zim_vit_l_2092` |
| `luciddreamer.yaml` | `thirdparty/LucidDreamer` | `stabilityai--stable-diffusion-2-inpainting` and any upstream HF cache entries under `huggingface/` |
| `hy_worldplay.yaml` | `thirdparty/HY-WorldPlay` | `HY-WorldPlay`, `HY-WorldPlay/ar_distilled_action_model/model.safetensors` |
| `wonderjourney.yaml` | `thirdparty/WonderJourney` | `WonderJourney/dpt_beit_large_512.pt` plus `sd2-community--stable-diffusion-2-inpainting` and any upstream HF cache entries under `huggingface/` |
| `wonderworld.yaml` | `thirdparty/WonderWorld` | `WonderWorld/repvit_sam.pt`, `sd2-community--stable-diffusion-2-inpainting`, `prs-eth--marigold-v1-0`, `prs-eth--marigold-normals-v0-1`, and any upstream HF cache entries under `huggingface/` |
| `yume.yaml` | `thirdparty/YUME` | `stdstu123--Yume-5B-720P`, `InternVL3-2B-Instruct` |
| `worldfm.yaml` | `thirdparty/worldfm` | `worldfm` plus companion `ZIM/zim_vit_l_2092` |
| `matrix_game_1.yaml` | `thirdparty/Matrix-Game-1` | `Matrix-Game-1.0` |
| `matrix_game_2.yaml` | `thirdparty/Matrix-Game/Matrix-Game-2` | `Skywork--Matrix-Game-2.0` |
| `matrix_game.yaml` | `thirdparty/Matrix-Game-3.0` | `Matrix-Game-3.0` |
| `hunyuan_gamecraft.yaml` | `thirdparty/Hunyuan-GameCraft-1.0` | `tencent--Hunyuan-GameCraft-1.0` |

Bundled benchmark runtime defaults:

- `benchmark.metric_runtime.depth_accuracy.model_name: depth-anything/Depth-Anything-V2-Small-hf`
- `benchmark.metric_runtime.depth_collision.checkpoint_path: ckpt/Video-Depth-Anything/metric_video_depth_anything_vits.pth`
- `benchmark.metric_runtime.met3r_consistency.backbone: mast3r`
- `benchmark.metric_runtime.met3r_consistency.feature_backbone_weights: facebookresearch/dino:main`
- `benchmark.metric_runtime.met3r_consistency.upsampler: bilinear`
- `benchmark.metric_runtime.reconstruction_consistency.model_name: depth-anything/DA3-GIANT`
- `benchmark.metric_runtime.temporal_calibration.checkpoint_path: ckpt/Pulse-of-Motion/vc_common_10_60fps.ckpt`
- `benchmark.metric_runtime.long_sequence.amt_checkpoint_path: ckpt/amt_model/amt-s.pth`
- `benchmark.metric_runtime.long_sequence.raft_model_path: ckpt/raft_model/models/raft-things.pth`
- `benchmark.metric_runtime.long_sequence.musiq_model_path: ckpt/musiq/musiq_spaq_ckpt-358bb6af.pth`
- suite-level optional runtimes: `benchmark.metric_runtime.fvmd.repo_root: thirdparty/FVMD-frechet-video-motion-distance`, `benchmark.metric_runtime.jedi.repo_root: thirdparty/JEDi`
- HF/Torch caches: `ckpt/huggingface`, `ckpt/huggingface/hub`, `ckpt/transformers`, `ckpt/torch`

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

For `InSpatio-World`, the benchmark adapter uses the upstream `run_test_pipeline.sh` but skips Florence captioning and injects the benchmark prompt directly through a synthetic single-video `new.json`. The main knobs stay under `model.generation`: `traj_txt_path`, `checkpoint_path`, `da3_model_path`, `step2_gpus`, `step3_gpus`, `step3_nproc`, `master_port`, `relative_to_source`, `rotation_only`, `adaptive_frame`, `freeze_repeat`, `use_tae`, `compile_dit`, and `radius_ratio`. Relative checkpoint-like fields are resolved inside `ckpt/inspatio--world`.

For `lingbot-world`, the `model.generation` block is intended to stay close to the upstream CLI. Beyond the core fields (`task`, `size`, `frame_num`, `sample_solver`), the adapter also forwards controls such as `sample_steps`, `sample_shift`, `sample_guide_scale`, `ulysses_size`, `offload_model`, `t5_fsdp`, `t5_cpu`, `dit_fsdp`, `use_prompt_extend`, `prompt_extend_*`, `action_string_by_suite`, and `allow_act2cam`. The bundled configs currently cover the base camera checkpoint, the action-preview checkpoint, and the fast runtime variant. `generation.fast_checkpoint_dir` and `generation.shared_checkpoint_dir` still exist, but they now only accept explicit paths under `ckpt/`; the adapter materializes the upstream runtime layout from those explicit local directories. When a sample carries a usable `annotation_path`, the adapter maps it to upstream `--action_path` automatically and normalizes the benchmark pose/intrinsics layout into the format expected by `lingbot-world`.

For `GEN3C`, the benchmark adapter runs the upstream `gen3c_single_image.py` entrypoint against the cached conditioning frame and stages the canonical `ckpt/` layout for the repo. The main knobs under `model.generation` are `num_gpus`, `num_video_frames`, `height`, `width`, `fps`, `seed`, `guidance`, `num_steps`, `trajectory`, `camera_rotation`, `movement_distance`, `filter_points_threshold`, and the memory-offload flags mirrored from the upstream CLI.

For `Infinite-World`, the `model.generation` block keeps the benchmark-side knobs small: `config_path`, `bucket_config_name`, `num_chunks`, `num_sampling_steps`, `sample_shift`, `sample_guide_scale`, `fps`, and an `action_plan` that expands suite-specific symbolic interactions like `forward`, `forward_right`, and `camera_left` into the framewise move/view JSON expected by the upstream model.

For `NeoVerse`, the adapter supports both bundled video suites but uses different defaults by suite: `video_static` runs from the cached first frame plus an annotation-derived camera trajectory, while `video_dynamic` uses the reference video as the source sequence and keeps the original camera path unless you override `model.generation`. The main knobs are `input_mode`, `trajectory_source`, `trajectory`, `traj_mode`, `zoom_ratio`, `num_frames`, `height`, `width`, `alpha_threshold`, `low_vram`, and the checkpoint layout fields `model_subdir` / `reconstructor_filename`.

For `HunyuanWorld`, the adapter targets static suites only. It uses the reference image or first video frame to run `demo_panogen.py`, feeds the resulting `panorama.png` into `demo_scenegen.py`, and archives the generated `mesh_layer*.ply` files and intermediates into a single `.zip`. This is generation-only plumbing for now: the current `evaluate` path does not score those mesh archives. All runtime caches now resolve under `ckpt/`, and the upstream ZIM checkpoint is staged from `ckpt/ZIM/zim_vit_l_2092` into the expected source-relative location at launch time.

For `LucidDreamer`, the adapter targets static suites only. It runs the upstream `run.py` CLI against the cached reference image or first video frame, stages the rendered `campath_render`.mp4 back into the benchmark prediction path, and preserves the generated `gsplat.ply` plus depth video as sidecars next to the prediction. Any upstream HF downloads must already be present under `ckpt/`, because the runtime now runs in offline mode with caches pinned to `ckpt/huggingface` and `ckpt/torch`.

For `HY-WorldPlay`, the bundled config expects a local repo under `thirdparty/HY-WorldPlay`, a prepared local weights tree under `checkpoint_dir`, and the released action checkpoint under `model.generation.action_ckpt`. The adapter converts benchmark `poses.npy` / `intrinsics.npy` pairs, or image-suite `camera_path` prompts when no pose sidecar exists, into the pose JSON consumed by `hyvideo/generate.py`, then stages the generated `gen.mp4` or `gen_sr.mp4` back into the benchmark prediction path.

For `WonderJourney`, the adapter targets static suites and keeps the benchmark output contract: the benchmark prediction is an `.mp4` rendered from generated frames, not a zip archive. It materializes a benchmark conditioning frame plus prompt into `examples/examples.yaml`, stages the MiDaS `dpt_beit_large_512.pt` checkpoint into the temporary run directory, runs upstream `run.py`, and copies the merged `output.mp4` back as the prediction with optional reverse/keyframe sidecars. Upstream HF loads are expected to resolve from `ckpt/huggingface` or explicit paths under `ckpt/`.

For `WonderWorld`, the adapter targets static suites through a noninteractive batch runner. It uses benchmark `poses.npy` / `intrinsics.npy`, or synthesizes image-suite camera poses from the prompt contract, renders `all_interpframes`, writes the benchmark prediction as `.mp4`, and only stores generated frames plus 3DGS files in an optional `.sidecar.zip` archive. The explicit RepViT checkpoint now lives under `ckpt/WonderWorld`, and any remaining upstream HF cache lookups are pinned to `ckpt/huggingface`.

For `WorldFM`, the adapter converts each sample's `poses.npy` and `intrinsics.npy` into the upstream `meta.json` format and then runs `run_pipeline.py` against the cached first frame. The bundled config intentionally stays narrow: `model_filename`, `vae_subdir`, `step`, `image_size`, `cfg_scale`, `render_size`, `fps`, and `gpu_index`. It is registered for `video_static` only because the upstream model is a single-image, pose-driven novel-view generator rather than a dynamic scene model. The runner also stages the upstream ZIM checkpoint from `ckpt/ZIM/zim_vit_l_2092` into the expected source-relative location before execution.

For `YUME`, the adapter imports the upstream `webapp_single_gpu.py` entrypoint directly inside a subprocess, rewires its checkpoint paths, and calls `LongGenArgs` / `long_generate()` without starting the Flask server. The bundled config defaults to a single-segment `I2V` pass from the cached first frame and exposes the core upstream knobs under `model.generation`: `sample_steps`, `sample_num`, `frame_zero`, `shift`, `resolution`, optional `caption_model_dir`, and suite-specific `camera_movement*_by_suite` overrides. Both the base model and the optional caption model must now be prepared under `ckpt/`.

For `FlashWorld`, the adapter converts each sample's `poses.npy` and `intrinsics.npy` into the
upstream JSON camera trajectory format and then runs `cli.py` against the cached first frame plus
the benchmark prompt. The main knobs stay narrow: `num_frames`, `image_height`, `image_width`,
`image_index`, `video_fps`, the three offload flags, and optional `export_ply` / `export_spz`
sidecars. It is currently registered for `video_static` only because the upstream model is a
single-image static-scene generator rather than a dynamic world model.

## Dataset Analytics And Docs

Refresh the dataset analytics pages:

```bash
worldarena build-docs --config config/datasets.yaml
```

Preview the documentation site locally:

```bash
mkdocs serve
```

Additional documentation:

- `docs/architecture.md`: repo boundaries, task families, unified model config, and API integration pattern
- `docs/benchmark/`: benchmark scope, metric policy, reporting contract, and per-metric notes
- `docs/thirdparty.md`: third-party repos, cache layout, and checkpoint integration
- `docs/development.md`: maintainer workflow, validation commands, release hygiene
