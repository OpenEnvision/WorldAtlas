from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Cosmos Framework batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    spec_path = Path(args.batch_spec_path).expanduser().resolve()
    cosmos_input = spec_path.with_suffix(".cosmos.jsonl")
    cosmos_output = spec_path.parent / f"{spec_path.stem}_cosmos_outputs"

    model_mode = str(generation.get("model_mode", "image2video"))
    fps = int(generation.get("fps", 24))
    with cosmos_input.open("w", encoding="utf-8") as file:
        for row in rows:
            sample_args = {
                "sample_name": str(row["prediction_stem"]),
                "model_mode": model_mode,
                "prompt": str(row["prompt"]),
                "fps": fps,
                "num_frames": int(generation.get("num_frames", 189)),
                "resolution": str(generation.get("resolution", "720")),
                "aspect_ratio": str(generation.get("aspect_ratio", "16,9")),
            }
            if model_mode in {"image2video", "video2video"}:
                sample_args["vision_path"] = str(row["conditioning_image"])
            file.write(json.dumps(sample_args, ensure_ascii=False) + "\n")

    checkpoint_path = str(generation.get("checkpoint_path") or args.checkpoint_dir or "Cosmos3-Nano")
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset",
        str(generation.get("parallelism_preset", "throughput")),
        "-i",
        str(cosmos_input),
        "-o",
        str(cosmos_output),
        "--checkpoint-path",
        checkpoint_path,
        "--model-mode",
        model_mode,
        "--seed",
        str(int(generation.get("seed", 0))),
    ]
    if generation.get("no_guardrails", False):
        command.append("--no-guardrails")
    subprocess.run(command, check=True, cwd=str(repo_root))

    failed = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        try:
            source = cosmos_output / str(row["prediction_stem"]) / "vision.mp4"
            copy_output(source, Path(str(row["output_path"])))
            print_status(sample_id, "generated", output_path=str(row["output_path"]), source=str(source))
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
