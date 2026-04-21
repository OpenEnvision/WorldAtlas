# Third-Party Repositories

Clone or symlink upstream model repositories here when you want to use the bundled adapters.

Expected layout:

```text
third_party/
  lingbot-world/
  WorldCam/
  Matrix-Game-3.0/
  metric-source/
```

The helper command can link existing checkouts into this directory:

```bash
uv run worldarena prepare-layout \
  --repo-link lingbot-world=/path/to/lingbot-world \
  --repo-link WorldCam=/path/to/WorldCam \
  --repo-link Matrix-Game-3.0=/path/to/Matrix-Game-3.0 \
  --repo-link metric-source=/path/to/compatible-metric-checkout
```

This repository does not vendor or redistribute upstream code. Users are responsible for cloning the correct upstream repos and accepting their licenses separately.
