#!/usr/bin/env python3
"""Run one GEN3C shard through WorldArena's checkpoint-resident adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _batch_input_payload(*, item: dict[str, Any], error_path: Path) -> dict[str, Any]:
    """Build one request understood by ``gen3c_persistent_runner``."""
    generation = dict(item.get("generation") or {})
    payload = {
        "prompt": str(item.get("prompt") or ""),
        "visual_input": str(Path(item["conditioning_image"]).expanduser().resolve()),
        "output_path": str(Path(item["prediction_path"]).expanduser().resolve()),
        "error_path": str(error_path.expanduser().resolve()),
    }
    for key in (
        "camera_path",
        "trajectory",
        "camera_rotation",
        "movement_distance",
        "filter_points_threshold",
        "foreground_masking",
        "negative_prompt",
        "num_video_frames",
        "height",
        "width",
        "fps",
    ):
        if generation.get(key) is not None:
            payload[key] = generation[key]
    return payload


def _output_signature(path: Path) -> tuple[int, int] | None:
    """Return a stable non-empty-file signature used by batch supervisors."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file() or stat.st_size <= 0:
        return None
    return stat.st_size, stat.st_mtime_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one persistent GEN3C benchmark shard.")
    parser.add_argument("--model-config", default="config/models/gen3c.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/benchmark.yaml")
    parser.add_argument("--suites", default="image_static")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--visible-devices", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-u",
        "-m",
        "worldarena.cli",
        "generate",
        "--config",
        args.config,
        "--model-config",
        args.model_config,
        "--manifest",
        args.manifest,
        "--output-dir",
        args.output_dir,
        "--suites",
        args.suites,
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    if args.fail_fast:
        command.append("--fail-fast")

    env = os.environ.copy()
    if args.visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)
    pythonpath = [str(project_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    completed = subprocess.run(command, cwd=project_root, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
