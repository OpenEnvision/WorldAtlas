#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=${OPEN_DREAMER_REPO_ROOT:-$PROJECT_ROOT/thirdparty/open-dreamer}
CHECKPOINT_ROOT=${OPEN_DREAMER_CHECKPOINT_ROOT:-$PROJECT_ROOT/ckpt/open-dreamer}
UPSTREAM_REVISION=${OPEN_DREAMER_UPSTREAM_REVISION:-5a4127fcbbd37b3cd9237904d645fb11348bab28}
CHECKPOINT_REPO=${OPEN_DREAMER_CHECKPOINT_REPO:-reactor-team/open-dreamer}
CHECKPOINT_REVISION=${OPEN_DREAMER_CHECKPOINT_REVISION:-main}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_ENDPOINT
export HF_HUB_DISABLE_XET=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  git -C "$PROJECT_ROOT" submodule update --init thirdparty/open-dreamer
fi

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  git clone https://github.com/next-state/open-dreamer.git "$REPO_ROOT"
  git -C "$REPO_ROOT" checkout --detach "$UPSTREAM_REVISION"
elif [[ $(git -C "$REPO_ROOT" rev-parse HEAD) != "$UPSTREAM_REVISION" ]]; then
  echo "WARNING: existing Open Dreamer checkout is not the integrated revision: $UPSTREAM_REVISION" >&2
fi

# Upstream pins Python 3.11 and jax[cuda12] through uv; the lockfile is the only
# tested combination, so build the virtualenv with uv rather than plain pip.
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install uv
fi
(cd "$REPO_ROOT" && uv sync)

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: uv sync did not create $VENV_PYTHON" >&2
  exit 1
fi

# The WorldArena runner reads the conditioning clip with imageio and writes the
# prediction with the ffmpeg plugin, which upstream only needs for training vis.
"$VENV_PYTHON" -m pip install --quiet "imageio[ffmpeg]>=2.31"

mkdir -p "$CHECKPOINT_ROOT"
if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
else
  HF_CLI=("$VENV_PYTHON" -m huggingface_hub.commands.huggingface_cli download)
  "$VENV_PYTHON" -m pip install --quiet "huggingface_hub[cli]"
fi
"${HF_CLI[@]}" "$CHECKPOINT_REPO" \
  --revision "$CHECKPOINT_REVISION" \
  --local-dir "$CHECKPOINT_ROOT"

# Fail loudly here rather than mid-batch if the Orbax layout is not what the
# rollout expects.
PYTHONPATH="$PROJECT_ROOT" "$VENV_PYTHON" - "$CHECKPOINT_ROOT" <<'PY'
from pathlib import Path
import sys

from worldarena.models.adapters.open_dreamer_batch_runner import resolve_checkpoint_layout

layout = resolve_checkpoint_layout(Path(sys.argv[1]), use_ema=True)
print(f"Resolved Open Dreamer checkpoint step {layout['step']} at {layout['step_dir']}")
PY

PYTHONPATH="$PROJECT_ROOT" "$VENV_PYTHON" -m \
  worldarena.models.adapters.open_dreamer_batch_runner --help >/dev/null
PYTHONPATH="$REPO_ROOT" "$VENV_PYTHON" -c "import dreamer.generation, dreamer.checkpointing" >/dev/null

echo "Open Dreamer is ready."
echo "Repository:  $REPO_ROOT"
echo "Environment: $REPO_ROOT/.venv"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo
echo "Verify JAX sees a GPU before generating:"
echo "  $VENV_PYTHON -c \"import jax; print(jax.devices('gpu'))\""
