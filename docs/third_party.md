# Runtime Setup

WorldArena is meant to be publishable as a benchmark repository without vendoring large model repos or checkpoints. The runtime layout therefore separates three concerns:

- benchmark code and docs live in this repository
- upstream model repos live under `third_party/` or are symlinked there
- reusable checkpoints and reference-metric assets live under `cache/hfd/` or are symlinked there

## Directory Contract

Default layout:

```text
WorldArena/
  cache/
    hfd/
  third_party/
    lingbot-world/
    WorldCam/
    Matrix-Game-3.0/
    metric-source/
```

The bundled model configs assume this layout by default:

- `config/models/lingbot_world.yaml`
  - `repo_root: third_party/lingbot-world`
  - `checkpoint_dir: cache/hfd/robbyant--lingbot-world-base-cam`
- `config/models/worldcam.yaml`
  - `repo_root: third_party/WorldCam`
  - `checkpoint_dir: cache/hfd/worldcam--worldcam`
- `config/models/matrix_game.yaml`
  - `repo_root: third_party/Matrix-Game-3.0`
  - `checkpoint_dir: cache/hfd/Matrix-Game-3.0`

If your machine uses a different layout, either edit a local copy of the YAML config or keep the repo layout stable and point it at shared resources with symlinks.

## Helper Command

Create the local layout:

```bash
uv run worldarena prepare-layout
```

Point `cache/hfd` at an existing shared cache:

```bash
uv run worldarena prepare-layout --shared-hf-cache /path/to/shared/hf-cache
```

Link existing upstream repos into `third_party/`:

```bash
uv run worldarena prepare-layout \
  --repo-link lingbot-world=/path/to/lingbot-world \
  --repo-link WorldCam=/path/to/WorldCam \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0
```

Link a compatible metric-source checkout the same way:

```bash
uv run worldarena prepare-layout \
  --repo-link metric-source=/path/to/compatible-metric-checkout
```

You can combine both forms in a single command. The helper prints a JSON summary including:

- the effective `cache/hfd` path
- recommended `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` values
- linked third-party repos
- the expected adapter layout for the bundled model configs

Use `--force` only when you intentionally want to replace an existing `cache/hfd` or `third_party/<name>` path.

## Environment Profiles

Core benchmark and dataset analytics:

```bash
uv sync --python 3.10
```

Official metric backends:

```bash
uv sync --python 3.10 --extra eval --extra quality --extra video-metrics
```

That profile is expected to cover `cvxpy`, `iopath`, `pycocotools`, and the rest of the heavy runtime stack. For `motion_smoothness`, some machines still need the exact `causal-conv1d` / `mamba-ssm` pair installed as recommended by the compatible metric-source checkout.

Integrated world-model adapters:

```bash
uv sync --python 3.10 --group world_models --extra eval --extra quality
```

The heavy metric stack and the world-model stack are optional. You only need them if you want to run official metrics locally or execute the bundled model adapters.

If multiple compatible metric-source checkouts exist on the machine, pin the one you want with:

```bash
export WORLDARENA_METRIC_SOURCE_ROOT=$PWD/third_party/metric-source
```

Then verify the result with:

```bash
uv run worldarena check-env
```

This probe stays read-mostly at the environment layer. It does not execute the heavy official metrics; it only checks that the source tree, Python packages, and declared metric entrypoints line up with the benchmark contract.

## Checkpoints And Cache Policy

`cache/hfd` is treated as a machine-local cache root. It may contain:

- Hugging Face hub downloads used by metric backends and adapters
- manually placed checkpoints whose upstream projects expect a fixed local path
- symlinks to larger centralized cache volumes on shared infrastructure

Recommended environment variables:

```bash
export HF_HOME=$PWD/cache/hfd
export HF_HUB_CACHE=$PWD/cache/hfd/huggingface/hub
export TRANSFORMERS_CACHE=$PWD/cache/hfd/transformers
```

Reference metric backends can reuse the same cache root as long as their checkpoints resolve under that tree.

The official runtime adapter only imports the specific heavy metric modules it needs. That keeps the benchmark repo itself compact while still reusing the reference algorithms and checkpoint paths.

## Third-Party Repository Policy

`third_party/` is for local checkouts or symlinks only. This repository does not vendor those repos.

Practical rules:

- keep the clone name aligned with the config default when possible
- pin upstream commits in your experiment logs or local notes, not by committing their code here
- accept the upstream license separately for each repo and model weight
- never commit cloned repos or their generated artifacts back into WorldArena

## Optional Separate Runtime Python

Some upstream model repos run best with a CUDA-enabled Python outside the main `uv` environment. The bundled configs support `python_bin: auto`, which prefers `.runtime312/bin/python` when it exists.

Example bootstrap:

```bash
uv venv --python 3.12 --system-site-packages .runtime312
uv pip install --python .runtime312/bin/python \
  diffusers decord dashscope easydict scipy trimesh imageio imageio-ffmpeg \
  modelscope sentencepiece protobuf pynvml opencv-python==4.10.0.84
```

Keep benchmark orchestration in the repo environment and isolate adapter-specific runtime packages in `.runtime312` when necessary.
