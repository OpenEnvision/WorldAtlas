# Model configurations

The public integrations are scoped to models appearing in at least one of the
published image-static, image-dynamic, 60-second memory or 30-second memory
leaderboards (snapshot checked on 2026-10-08). Unlisted model families and their
standalone configurations are excluded. Shared runtime and metric dependencies
remain where a retained model needs them.

A published score does not imply an in-tree generation adapter: AC3D, Astra and
CogVideo results were imported, and their generation integrations are not bundled.
Generic adapter contracts and the HTTP API template remain available for adding a
new model. Configuration availability is not a claim of GPU reproduction.

Each model family has its own directory, matching the grouping in
`worldatlas/models/adapters/`. Variants such as Base/Fast, checkpoint sizes,
and Memory durations stay together and keep their existing filenames.

```text
config/models/
├── lingbot_world/
│   ├── lingbot_world.yaml
│   ├── lingbot_world_fast.yaml
│   ├── lingbot_world_fast_memory.yaml
│   └── lingbot_world_fast_memory_30s.yaml
├── solarwm/
│   ├── solarwm.yaml
│   └── solarwm_h3.yaml
└── worldcrafter/
    ├── worldcrafter_fast.yaml
    └── worldcrafter_fast_memory.yaml
```

Pass the grouped path to commands, for example
`--model-config config/models/lingbot_world/lingbot_world_fast.yaml`.
API configuration templates (`*.yaml.example`) also live in their model directory.

The model config loader still accepts old flat paths such as
`config/models/lingbot_world_fast.yaml` when the filename uniquely identifies a
config. Existing files, including frozen job snapshots, take precedence.
New scripts should use the grouped paths; Python consumers of roster basenames
can use `worldatlas.models.config.find_model_config(config_dir, filename)`.

Public configurations use `python_bin: python` from the active environment,
`thirdparty/<name>` for source, and `ckpt/<name>` for weights. Install the selected
upstream environment before generation. To keep inference separate from metric
evaluation, copy the configuration to `<name>_local.yaml` and set `python_bin` to
that inference environment's absolute interpreter path. Local configurations
are ignored by Git and excluded from source distributions.

Memory variants are shipped as complete YAML files and can be edited directly.
Their durations, source pools and trajectory settings must match the evaluation
protocol. When enumerating public model configs, use `rglob("*.yaml")` and exclude
local/debug overrides. See [Getting Started](https://worldatlas.worldatlas-site.workers.dev/doc/getting-started/) for a complete run.
