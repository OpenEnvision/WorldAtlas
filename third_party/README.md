# Third-Party Repositories

Clone or symlink upstream model repositories here when you want to use the bundled adapters.

Expected layout:

```text
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

The helper command can link existing checkouts into this directory:

```bash
uv run worldarena prepare-layout \
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

`metric-source` is used for the official video metric stack. `Pulse-of-Motion` is used for the temporal calibration metrics.

`Pulse-of-Motion` may auto-download its upstream checkpoint on first use, so the checkout can be linked before `inference/ckpts/vc_common_10_60fps.ckpt` exists locally.

This repository does not vendor or redistribute upstream code. Users are responsible for cloning the correct upstream repos and accepting their licenses separately.
