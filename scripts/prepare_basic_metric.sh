#!/usr/bin/env bash
# Prepare everything the 5 new models need for basic-metric (image_static +
# image_dynamic) inference:
#   (a) link WorldAtlas image data      -> WorldArena/data/image
#   (b) link upstream repos             -> WorldArena/thirdparty/<repo>
#   (c) link model weights              -> WorldArena/ckpt/<name>
#   (d) link conda envs                 -> <workspace>/conda/envs/<env>
#   (e) build the 5000-image manifest via `worldarena build-manifest`
#
# Idempotent: existing correct symlinks are left untouched; a real (non-symlink)
# directory at a target aborts the script instead of being overwritten.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_ROOT="${WORLD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"          # .../arena/WorldArena
ARENA_ROOT="$(cd -- "$WORLD_ROOT/.." && pwd)"                        # .../arena
CONDA_ENVS_DIR="$ARENA_ROOT/conda/envs"                             # matches config python_bin (../conda/envs/<env>)

BENCHMARK_CONFIG="${WORLDARENA_BATCH_BENCHMARK_CONFIG:-config/benchmark.yaml}"
# build-manifest needs the worldarena runtime deps (av, pandas, omegaconf, ...).
# There is no dedicated 'worldarena' env here; envs/alayaworld carries them, so
# use it by default. Override with PREPARE_PYTHON_BIN if you have a proper env.
DEFAULT_MANIFEST_PYTHON="$ARENA_ROOT/envs/alayaworld/bin/python"
if [[ ! -x "$DEFAULT_MANIFEST_PYTHON" ]]; then
    DEFAULT_MANIFEST_PYTHON="python"
fi
PYTHON_BIN="${PREPARE_PYTHON_BIN:-$DEFAULT_MANIFEST_PYTHON}"
SKIP_MANIFEST="${PREPARE_SKIP_MANIFEST:-0}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"

cd "$WORLD_ROOT"

# link_dir SRC DST : create/refresh symlink DST -> SRC (idempotent, non-destructive).
link_dir() {
    local src="$1" dst="$2"
    if [[ ! -e "$src" ]]; then
        echo "  SKIP (source missing): $src" >&2
        return 1
    fi
    if [[ -L "$dst" ]]; then
        local current
        current="$(readlink -f "$dst" 2>/dev/null || true)"
        if [[ "$current" == "$(readlink -f "$src")" ]]; then
            echo "  ok (already linked): $dst -> $src"
            return 0
        fi
        echo "  relinking: $dst -> $src (was $current)"
        ln -sfn "$src" "$dst"
        return 0
    fi
    if [[ -e "$dst" ]]; then
        echo "ERROR: $dst exists as a real path (not a symlink); refusing to overwrite." >&2
        return 2
    fi
    mkdir -p "$(dirname "$dst")"
    ln -sfn "$src" "$dst"
    echo "  linked: $dst -> $src"
}

echo "[prepare] (a) data ------------------------------------------------------"
mkdir -p "$WORLD_ROOT/data"
link_dir "$ARENA_ROOT/data/WorldAtlas/image" "$WORLD_ROOT/data/image"
# build-manifest also scans video_root; keep an empty dir so rglob finds nothing
# instead of the whole tree being absent.
mkdir -p "$WORLD_ROOT/data/video"

echo "[prepare] (b) upstream repos -> thirdparty/ -----------------------------"
mkdir -p "$WORLD_ROOT/thirdparty"
link_dir "$ARENA_ROOT/model/ABot-World"       "$WORLD_ROOT/thirdparty/ABot-World"       || true
link_dir "$ARENA_ROOT/model/AlayaWorld"       "$WORLD_ROOT/thirdparty/AlayaWorld"       || true
link_dir "$ARENA_ROOT/model/Matrix-Game-3.5"  "$WORLD_ROOT/thirdparty/Matrix-Game-3.5"  || true
link_dir "$ARENA_ROOT/model/lingbot-world"    "$WORLD_ROOT/thirdparty/lingbot-world"    || true
link_dir "$ARENA_ROOT/model/lingbot-world-v2" "$WORLD_ROOT/thirdparty/lingbot-world-v2" || true
link_dir "$ARENA_ROOT/model/lingbot-video"    "$WORLD_ROOT/thirdparty/lingbot-video"    || true

echo "[prepare] (c) checkpoints -> ckpt/ --------------------------------------"
mkdir -p "$WORLD_ROOT/ckpt"
link_dir "$ARENA_ROOT/world_models/abot-world/checkpoints/ABot-World-0-5B-LF"        "$WORLD_ROOT/ckpt/ABot-World-0-5B-LF"            || true
link_dir "$ARENA_ROOT/world_models/alaya-world/checkpoints"                          "$WORLD_ROOT/ckpt/AlayaWorld"                    || true
link_dir "$ARENA_ROOT/world_models/matrix-game-3.5/checkpoints"                      "$WORLD_ROOT/ckpt/Matrix-Game-3.5"               || true
link_dir "$ARENA_ROOT/world_models/lingbot-world/lingbot-world-base-cam"             "$WORLD_ROOT/ckpt/lingbot-world-base-cam"        || true
link_dir "$ARENA_ROOT/world_models/lingbot-world-v2/lingbot-world-v2-14b-causal-fast" "$WORLD_ROOT/ckpt/lingbot-world-v2-14b-causal-fast" || true
link_dir "$ARENA_ROOT/world_models/lingbot-video/lingbot-video-moe-30b-a3b"          "$WORLD_ROOT/ckpt/lingbot-video-moe-30b-a3b"     || true
# Dense variant is optional (may be absent if only MoE weights were downloaded).
link_dir "$ARENA_ROOT/world_models/lingbot-video/lingbot-video-dense-1.3b"           "$WORLD_ROOT/ckpt/lingbot-video-dense-1.3b"      || true

echo "[prepare] (d) conda envs -> $CONDA_ENVS_DIR -----------------------------"
mkdir -p "$CONDA_ENVS_DIR"
# Configs whose python_bin is ../conda/envs/<env>/bin/python.
for env_name in lingbot-world lingbot-world-v2 alayaworld matrix-game-3.5; do
    link_dir "$ARENA_ROOT/envs/$env_name" "$CONDA_ENVS_DIR/$env_name" || true
done
# ABot-World and LingBot-Video configs expect thirdparty/<repo>/.venv. There is
# no matching entry under envs/, so warn rather than invent a link.
for repo_venv in "thirdparty/ABot-World/.venv" "thirdparty/lingbot-video/.venv"; do
    if [[ ! -e "$WORLD_ROOT/$repo_venv" ]]; then
        echo "  WARNING: $repo_venv missing; prepare its Python env manually (config python_bin points here)." >&2
    fi
done
# The CLI/orchestrator itself needs a 'worldarena' env; launch scripts default to
# ../conda/envs/worldarena/bin/python and fall back to 'python' if absent.
if [[ ! -e "$CONDA_ENVS_DIR/worldarena" ]]; then
    echo "  NOTE: no worldarena env under $CONDA_ENVS_DIR; launchers will use the active 'python'." >&2
fi

if [[ "$SKIP_MANIFEST" == "1" || "$SKIP_MANIFEST" == "true" ]]; then
    echo "[prepare] (e) build-manifest SKIPPED (PREPARE_SKIP_MANIFEST=1)"
    echo "[prepare] done."
    exit 0
fi

echo "[prepare] (e) build-manifest --------------------------------------------"
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m worldarena.cli build-manifest --config "$BENCHMARK_CONFIG"

MANIFEST_PATH="$WORLD_ROOT/artifacts/benchmark/manifest/worldarena_manifest.jsonl"
if [[ -f "$MANIFEST_PATH" ]]; then
    # The existing launch_*_8gpu.sh scripts default to a dated snapshot name.
    # Alias it to the freshly built manifest so those defaults work unchanged.
    DATED_ALIAS="$WORLD_ROOT/artifacts/benchmark/manifest/worldarena_manifest_local_data_20260722.jsonl"
    if [[ ! -e "$DATED_ALIAS" || -L "$DATED_ALIAS" ]]; then
        ln -sfn "worldarena_manifest.jsonl" "$DATED_ALIAS"
        echo "[prepare] aliased dated manifest name -> worldarena_manifest.jsonl"
    fi
    echo "[prepare] manifest rows: $(wc -l < "$MANIFEST_PATH")  ($MANIFEST_PATH)"
    echo "[prepare] per-suite counts:"
    "$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import json, sys
from collections import Counter
counts = Counter()
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            counts[json.loads(line).get("suite", "?")] += 1
for suite, n in sorted(counts.items()):
    print(f"    {suite}: {n}")
PY
else
    echo "ERROR: manifest not produced at $MANIFEST_PATH" >&2
    exit 1
fi

echo "[prepare] done."
