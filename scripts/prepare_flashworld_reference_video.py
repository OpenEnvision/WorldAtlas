from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an original reference video at FlashWorld's official comparison scale."
    )
    parser.add_argument("--model-config", default="config/models/flashworld.yaml", type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--num-frames", default=None, type=int)
    parser.add_argument("--video-fps", default=None, type=int)
    parser.add_argument("--image-height", default=None, type=int)
    parser.add_argument("--image-width", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    args = parse_args()
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.models.adapters.flashworld_runner import (
        flashworld_render_frame_count,
        write_flashworld_aligned_reference_video,
    )
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() not in {"flashworld", "flash_world", "flash-world"}:
        raise ValueError(f"unsupported model family for FlashWorld reference prep: {model_config.family}")

    output_path = args.output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"aligned reference already exists: {output_path}")

    generation = dict(model_config.generation)
    num_frames = int(args.num_frames or generation.get("num_frames", 24))
    video_fps = int(args.video_fps or generation.get("video_fps", 15))
    image_height = int(args.image_height or generation.get("image_height", 480))
    image_width = int(args.image_width or generation.get("image_width", 704))
    target_frame_count = flashworld_render_frame_count(
        num_frames=num_frames,
        video_fps=video_fps,
    )

    summary = write_flashworld_aligned_reference_video(
        input_video=args.input_video,
        output_path=output_path,
        image_height=image_height,
        image_width=image_width,
        target_frame_count=target_frame_count,
        video_fps=video_fps,
    )
    summary["model_config"] = str(args.model_config.resolve())
    summary["num_frames"] = num_frames
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
