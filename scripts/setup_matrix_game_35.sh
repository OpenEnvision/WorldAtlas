#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd -- "$PROJECT_ROOT/.." && pwd)
ENV_PREFIX=${MATRIX_GAME_35_ENV_PREFIX:-$WORKSPACE_ROOT/conda/envs/matrix-game-3.5}
REPO_ROOT=${MATRIX_GAME_35_REPO_ROOT:-$PROJECT_ROOT/thirdparty/Matrix-Game-3.5}
CHECKPOINT_ROOT=${MATRIX_GAME_35_CHECKPOINT_ROOT:-$PROJECT_ROOT/ckpt/Matrix-Game-3.5}
UPSTREAM_REVISION=${MATRIX_GAME_35_UPSTREAM_REVISION:-6c94dd787659aa19fac22581cd8bea54e65d813f}
MATRIX_CHECKPOINT_REVISION=${MATRIX_GAME_35_CHECKPOINT_REVISION:-40c172355efa32d4e2a44076569e310807788f8a}
WAN_CHECKPOINT_REVISION=${MATRIX_GAME_35_WAN_REVISION:-921dbaf3f1674a56f47e83fb80a34bac8a8f203e}
DA3_CHECKPOINT_REVISION=${MATRIX_GAME_35_DA3_REVISION:-b2359bdf726fb44ef62acca04d629dcf158053e7}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PYPI_INDEX_URL=${MATRIX_GAME_35_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}
export HF_ENDPOINT
export HF_HUB_DISABLE_XET=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  git -C "$PROJECT_ROOT" submodule update --init thirdparty/Matrix-Game-3.5
fi

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  git clone https://github.com/Riemann-Dynamics/Matrix-Game-3.5.git "$REPO_ROOT"
  git -C "$REPO_ROOT" checkout --detach "$UPSTREAM_REVISION"
elif [[ $(git -C "$REPO_ROOT" rev-parse HEAD) != "$UPSTREAM_REVISION" ]]; then
  echo "WARNING: existing Matrix-Game 3.5 checkout is not the integrated revision: $UPSTREAM_REVISION" >&2
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -p "$ENV_PREFIX" python=3.10 pip -y
fi

"$ENV_PREFIX/bin/python" -m pip install --index-url "$PYPI_INDEX_URL" --upgrade pip
"$ENV_PREFIX/bin/python" -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url "$PYPI_INDEX_URL"
"$ENV_PREFIX/bin/python" -m pip install --index-url "$PYPI_INDEX_URL" \
  -r "$REPO_ROOT/requirements.txt"

# Make the vendored DA3 package importable before Matrix-Game's frustum module
# performs its import-time backend probe.
"$ENV_PREFIX/bin/python" - "$REPO_ROOT" <<'PY'
from pathlib import Path
import site
import sys

repo_root = Path(sys.argv[1]).resolve()
pth_path = Path(site.getsitepackages()[0]) / "matrix_game_35_da3.pth"
pth_path.write_text(
    f"{repo_root / 'third_party' / 'depth-anything-3' / 'src'}\n",
    encoding="utf-8",
)
PY

mkdir -p "$CHECKPOINT_ROOT"
"$ENV_PREFIX/bin/hf" download RiemannDynamics/Matrix-Game-3.5-Base \
  --revision "$MATRIX_CHECKPOINT_REVISION" \
  --local-dir "$CHECKPOINT_ROOT/Matrix-Game-3.5-Base"
"$ENV_PREFIX/bin/hf" download Wan-AI/Wan2.2-TI2V-5B \
  --revision "$WAN_CHECKPOINT_REVISION" \
  --exclude 'assets/*' --exclude 'examples/*' \
  --local-dir "$CHECKPOINT_ROOT/Wan2.2-TI2V-5B"
"$ENV_PREFIX/bin/hf" download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --revision "$DA3_CHECKPOINT_REVISION" \
  --local-dir "$CHECKPOINT_ROOT/DA3NESTED-GIANT-LARGE-1.1"

PYTHONPATH="$PROJECT_ROOT" "$ENV_PREFIX/bin/python" -m \
  worldarena.models.adapters.matrix_game_35_batch_runner --help >/dev/null
"$ENV_PREFIX/bin/python" "$REPO_ROOT/infer.py" --help >/dev/null

echo "Matrix-Game 3.5 is ready."
echo "Environment: $ENV_PREFIX"
echo "Repository:  $REPO_ROOT"
echo "Checkpoints: $CHECKPOINT_ROOT"
