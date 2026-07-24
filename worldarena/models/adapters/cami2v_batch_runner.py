"""Persistent batch runner for the official ZGCTroy/CamI2V demo API."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import yaml

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)
from worldarena.models.adapters.camera_i2v_common import (
    as_bool,
    camera_path_from_row,
    resolve_runtime_path,
    working_directory,
    write_official_camera_file,
)


MODEL_NAME = "worldarena_cami2v_512x320_100k"
OFFICIAL_PROMPT_SUFFIX = (
    " dramatic camera movement, high quality, cinematic shot, photorealistic, "
    "detailed fur, smooth motion"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena CamI2V batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _runtime_paths(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    generation: dict[str, Any],
) -> tuple[Path, Path]:
    config_file = resolve_runtime_path(
        generation.get("config_file"),
        bases=[repo_root],
        default="configs/inference/003_cami2v_512x320.yaml",
    )
    checkpoint_bases = [path for path in (checkpoint_dir, repo_root / "ckpts") if path is not None]
    checkpoint_file = resolve_runtime_path(
        generation.get("checkpoint_file"),
        bases=checkpoint_bases,
        default="512_cami2v_100k.pt",
    )
    return config_file, checkpoint_file


def _validate_runtime(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    config_file: Path,
    checkpoint_file: Path,
    generation: dict[str, Any],
) -> None:
    if not repo_root.is_dir():
        raise FileNotFoundError(f"CamI2V repo_root not found: {repo_root}")
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"CamI2V checkpoint_dir not found: {checkpoint_dir}")
    if not config_file.is_file():
        raise FileNotFoundError(f"CamI2V inference config not found: {config_file}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"CamI2V checkpoint not found: {checkpoint_file}")
    expected_size = generation.get("checkpoint_size_bytes")
    if expected_size is not None and checkpoint_file.stat().st_size != int(expected_size):
        raise ValueError(
            f"CamI2V checkpoint size mismatch for {checkpoint_file}: "
            f"expected {int(expected_size)}, got {checkpoint_file.stat().st_size}"
        )
    video_length = int(generation.get("video_length", generation.get("num_frames", 16)))
    if video_length != 16:
        raise ValueError(
            "the released CamI2V checkpoint has temporal_length=16; "
            f"got video_length={video_length}"
        )


def _write_runtime_metadata(
    *,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
    work_root: Path,
    config_file: Path,
    checkpoint_file: Path,
) -> tuple[Path, Path, list[str]]:
    model_meta_path = work_root / "models.json"
    camera_meta_path = work_root / "camera_poses.json"
    runtime_config_path = work_root / "cami2v_inference.yaml"
    width = int(generation.get("width", 512))
    height = int(generation.get("height", 320))
    video_length = int(generation.get("video_length", generation.get("num_frames", 16)))

    # The official loader only reaches its non-strict fallback after reading
    # model.pretrained_checkpoint.  Point that field at the selected released
    # checkpoint as well, so a harmless key mismatch cannot fall through to the
    # repo-relative, unreleased ``ckpts/512_cami2v.pt`` path.
    runtime_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    runtime_config.setdefault("model", {})["pretrained_checkpoint"] = str(checkpoint_file)
    runtime_config_path.write_text(
        yaml.safe_dump(runtime_config, sort_keys=False),
        encoding="utf-8",
    )

    model_meta_path.write_text(
        json.dumps(
            {
                MODEL_NAME: {
                    "config_file": str(runtime_config_path),
                    "ckpt_path": str(checkpoint_file),
                    "width": width,
                    "height": height,
                }
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    camera_metadata: dict[str, str] = {}
    camera_keys: list[str] = []
    camera_dir = work_root / "camera_poses"
    for index, row in enumerate(rows):
        camera_key = f"worldarena_{index:06d}"
        camera_file = camera_dir / f"{camera_key}.txt"
        write_official_camera_file(
            camera_file,
            camera_path_from_row(row),
            target_frames=video_length,
            runtime=generation,
        )
        camera_metadata[camera_key] = str(camera_file)
        camera_keys.append(camera_key)
    camera_meta_path.write_text(
        json.dumps(camera_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_meta_path, camera_meta_path, camera_keys


def _disable_demo_prompt_suffix(module: Any) -> None:
    """Keep benchmark prompts exact instead of adding the demo's fixed fur cue."""
    processor_class = module.SingleImageForInference
    if getattr(processor_class, "_worldarena_exact_prompt", False):
        return
    original = processor_class.get_batch_input

    def get_batch_input(self, ref_img, caption, *args, **kwargs):
        caption_text = str(caption)
        if caption_text.endswith(OFFICIAL_PROMPT_SUFFIX):
            caption_text = caption_text[: -len(OFFICIAL_PROMPT_SUFFIX)]
        return original(self, ref_img, caption_text, *args, **kwargs)

    processor_class.get_batch_input = get_batch_input
    processor_class._worldarena_exact_prompt = True


def _load_official_runtime(
    *,
    repo_root: Path,
    result_dir: Path,
    model_meta_path: Path,
    camera_meta_path: Path,
    generation: dict[str, Any],
) -> Any:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module = importlib.import_module("demo.cami2v_test")
    if not as_bool(generation.get("append_official_prompt_suffix"), default=False):
        _disable_demo_prompt_suffix(module)
    return module.Image2Video(
        result_dir=str(result_dir),
        model_meta_path=str(model_meta_path),
        camera_pose_meta_path=str(camera_meta_path),
        return_camera_trace=False,
        video_length=int(generation.get("video_length", generation.get("num_frames", 16))),
        save_fps=int(generation.get("fps", generation.get("save_fps", 10))),
        device=str(generation.get("device", "cuda")),
    )


def _generate_one(
    *,
    runtime: Any,
    row: dict[str, Any],
    row_index: int,
    camera_key: str,
    generation: dict[str, Any],
) -> Path:
    import numpy as np
    from PIL import Image

    conditioning_image = Path(str(row["conditioning_image"])).expanduser().resolve()
    if not conditioning_image.is_file():
        raise FileNotFoundError(f"conditioning image not found: {conditioning_image}")
    with Image.open(conditioning_image) as image:
        reference = np.asarray(image.convert("RGB")).copy()

    generated = runtime.get_image(
        MODEL_NAME,
        reference,
        str(row.get("prompt", "")),
        str(
            generation.get(
                "negative_prompt",
                "Fast movement, jittery motion, abrupt transitions, distorted body, "
                "missing limbs, unnatural posture, blurry, cropped, extra limbs, bad "
                "anatomy, deformed, glitchy motion, artifacts.",
            )
        ),
        camera_key,
        trace_extract_ratio=float(generation.get("trace_extract_ratio", 1.0)),
        frame_stride=int(generation.get("frame_stride", 1)),
        steps=int(generation.get("steps", generation.get("num_inference_steps", 25))),
        trace_scale_factor=float(generation.get("trace_scale_factor", 1.0)),
        camera_cfg=float(generation.get("camera_cfg", 1.0)),
        cfg_scale=float(generation.get("cfg_scale", generation.get("guidance_scale", 7.5))),
        seed=int(generation.get("seed", 123)) + row_index,
        enable_camera_condition=as_bool(generation.get("enable_camera_condition"), default=True),
        eta=float(generation.get("eta", 1.0)),
    )
    generated_path = Path(str(generated[0] if isinstance(generated, (list, tuple)) else generated))
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    copy_output(generated_path, output_path)
    if generated_path.resolve() != output_path:
        generated_path.unlink(missing_ok=True)
    return output_path


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None
    )
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    rows = load_batch_spec(batch_spec_path)
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    config_file, checkpoint_file = _runtime_paths(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
    )

    if os.environ.get("WORLDARENA_CAMI2V_DRY_RUN"):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "repo_root": str(repo_root),
                    "config_file": str(config_file),
                    "checkpoint_file": str(checkpoint_file),
                    "num_samples": len(rows),
                    "video_length": int(
                        generation.get("video_length", generation.get("num_frames", 16))
                    ),
                    "sample_preview": {
                        "sample_id": rows[0].get("sample_id"),
                        "camera_path": camera_path_from_row(rows[0]),
                        "prompt": rows[0].get("prompt", ""),
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    _validate_runtime(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        generation=generation,
    )
    output_parents = {
        Path(str(row["output_path"])).expanduser().resolve().parent for row in rows
    }
    # Sharded benchmark launches share one prediction directory.  Isolate the
    # metadata and camera files by the UUID-bearing batch-spec stem so parallel
    # runners cannot overwrite one another's camera trajectories.
    work_root = next(iter(output_parents)) / "_cami2v_work" / batch_spec_path.stem
    result_dir = work_root / "official_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_meta_path, camera_meta_path, camera_keys = _write_runtime_metadata(
        rows=rows,
        generation=generation,
        work_root=work_root,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
    )

    with working_directory(repo_root):
        runtime = _load_official_runtime(
            repo_root=repo_root,
            result_dir=result_dir,
            model_meta_path=model_meta_path,
            camera_meta_path=camera_meta_path,
            generation=generation,
        )
        failed = 0
        for row_index, (row, camera_key) in enumerate(zip(rows, camera_keys, strict=True)):
            sample_id = str(row.get("sample_id", row_index))
            requested_output = Path(str(row["output_path"])).expanduser().resolve()
            if requested_output.is_file() and requested_output.stat().st_size > 0:
                print_status(
                    sample_id,
                    "skipped_existing",
                    output_path=str(requested_output),
                    reason="output appeared after batch planning",
                )
                continue
            try:
                output_path = _generate_one(
                    runtime=runtime,
                    row=row,
                    row_index=row_index,
                    camera_key=camera_key,
                    generation=generation,
                )
                print_status(
                    sample_id,
                    "generated",
                    output_path=str(output_path),
                    camera_path=camera_path_from_row(row),
                    control_source="worldarena_camera_path",
                )
            except Exception as exc:
                failed += 1
                print_status(
                    sample_id,
                    "failed",
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            finally:
                gc.collect()
    return 0 if failed < len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
