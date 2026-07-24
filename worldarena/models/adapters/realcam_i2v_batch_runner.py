"""Persistent batch runner for ZGCTroy/RealCam-I2V on CogVideoX1.5."""

from __future__ import annotations

import argparse
import gc
import importlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any

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


MODEL_NAME = "worldarena_realcam_i2v_cogvideox1_5_5b_50k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena RealCam-I2V batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _finetune_root(repo_root: Path, generation: dict[str, Any]) -> Path:
    return resolve_runtime_path(
        generation.get("finetune_root"),
        bases=[repo_root],
        default="finetune",
    )


def _runtime_paths(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    generation: dict[str, Any],
) -> tuple[Path, Path, Path]:
    finetune_root = _finetune_root(repo_root, generation)
    checkpoint_bases = [path for path in (checkpoint_dir, finetune_root / "checkpoints") if path]
    controlnet_checkpoint = resolve_runtime_path(
        generation.get("controlnet_checkpoint"),
        bases=checkpoint_bases,
        default="RealCam-I2V_CogVideoX1.5-5B_50k.pt",
    )
    pretrained_bases = [
        path
        for path in (
            checkpoint_dir,
            checkpoint_dir.parent if checkpoint_dir is not None else None,
            finetune_root / "pretrained",
        )
        if path is not None
    ]
    pretrained_model = resolve_runtime_path(
        generation.get("pretrained_model_path"),
        bases=pretrained_bases,
        default="CogVideoX1.5-5B-I2V",
    )
    return finetune_root, pretrained_model, controlnet_checkpoint


def _validate_runtime(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    finetune_root: Path,
    pretrained_model: Path,
    controlnet_checkpoint: Path,
    generation: dict[str, Any],
) -> None:
    if not repo_root.is_dir():
        raise FileNotFoundError(f"RealCam-I2V repo_root not found: {repo_root}")
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"RealCam-I2V checkpoint_dir not found: {checkpoint_dir}")
    if not (finetune_root / "image_to_video.py").is_file():
        raise FileNotFoundError(f"RealCam-I2V finetune runtime not found: {finetune_root}")
    if not pretrained_model.is_dir():
        raise FileNotFoundError(
            f"CogVideoX1.5-5B-I2V pretrained model not found: {pretrained_model}"
        )
    if not controlnet_checkpoint.is_file():
        raise FileNotFoundError(
            f"RealCam-I2V ControlNet-XS checkpoint not found: {controlnet_checkpoint}"
        )
    expected_size = generation.get("controlnet_checkpoint_size_bytes")
    if expected_size is not None and controlnet_checkpoint.stat().st_size != int(expected_size):
        raise ValueError(
            f"RealCam-I2V checkpoint size mismatch for {controlnet_checkpoint}: "
            f"expected {int(expected_size)}, got {controlnet_checkpoint.stat().st_size}"
        )
    video_length = int(generation.get("num_frames", generation.get("video_length", 81)))
    if video_length < 17 or (video_length - 1) % 16 != 0:
        raise ValueError(
            "RealCam-I2V/CogVideoX1.5 expects frame counts 16n+1 (17, 33, ...); "
            f"got {video_length}"
        )


def _write_runtime_metadata(
    *,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
    work_root: Path,
    pretrained_model: Path,
    controlnet_checkpoint: Path,
) -> tuple[Path, Path, list[str]]:
    model_meta_path = work_root / "models.json"
    camera_meta_path = work_root / "camera_poses.json"
    video_length = int(generation.get("num_frames", generation.get("video_length", 81)))

    model_meta_path.write_text(
        json.dumps(
            {
                MODEL_NAME: {
                    "pretrained_model_path": str(pretrained_model),
                    "controlnetxs_model_path": str(controlnet_checkpoint),
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


def _install_inference_models_namespace(finetune_root: Path) -> None:
    """Expose upstream camera modules without importing every training module.

    Upstream ``finetune/models/__init__.py`` eagerly imports every Python file
    under ``models/``, including trainers that require wandb, peft, pydantic,
    deepspeed, and dataset packages.  The released inference entry point only
    needs ``models.camera_controller``.  A normal namespace package keeps those
    official modules importable while avoiding unrelated training dependencies.
    """
    models_root = (finetune_root / "models").resolve()
    if not models_root.is_dir():
        raise FileNotFoundError(f"RealCam-I2V models package not found: {models_root}")

    for module_name in tuple(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]

    package = ModuleType("models")
    package.__package__ = "models"
    package.__path__ = [str(models_root)]
    package.__spec__ = ModuleSpec("models", loader=None, is_package=True)
    sys.modules["models"] = package


def _load_official_runtime(
    *,
    finetune_root: Path,
    result_dir: Path,
    model_meta_path: Path,
    camera_meta_path: Path,
    generation: dict[str, Any],
) -> Any:
    if str(finetune_root) not in sys.path:
        sys.path.insert(0, str(finetune_root))
    _install_inference_models_namespace(finetune_root)
    module = importlib.import_module("image_to_video")
    # PyTorch 2.6 changed torch.load's default to weights_only=True.  The
    # released DeepSpeed checkpoint includes a harmless builtins.set in its
    # metadata, so allowlist that type while keeping the safer loader enabled.
    module.torch.serialization.add_safe_globals([set])
    return module.Image2Video(
        result_dir=str(result_dir),
        model_meta_path=str(model_meta_path),
        camera_pose_meta_path=str(camera_meta_path),
        save_fps=int(generation.get("fps", generation.get("save_fps", 16))),
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

    video_shape = (
        int(generation.get("num_frames", generation.get("video_length", 81))),
        int(generation.get("height", 512)),
        int(generation.get("width", 896)),
    )
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
        preview_video=None,
        steps=int(generation.get("steps", generation.get("num_inference_steps", 25))),
        trace_extract_ratio=float(generation.get("trace_extract_ratio", 1.0)),
        trace_scale_factor=float(generation.get("trace_scale_factor", 1.0)),
        camera_cfg=float(generation.get("camera_cfg", 1.0)),
        text_cfg=float(generation.get("text_cfg", generation.get("guidance_scale", 5.0))),
        seed=int(generation.get("seed", 123)) + row_index,
        noise_shaping=as_bool(generation.get("noise_shaping"), default=False),
        noise_shaping_minimum_timesteps=int(
            generation.get("noise_shaping_minimum_timesteps", 900)
        ),
        video_shape=video_shape,
        resize_for_rectangle_crop=as_bool(
            generation.get("resize_for_rectangle_crop"), default=True
        ),
    )
    generated_path = Path(str(generated)).expanduser().resolve()
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    copy_output(generated_path, output_path)
    if generated_path != output_path:
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
    finetune_root, pretrained_model, controlnet_checkpoint = _runtime_paths(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
    )

    if os.environ.get("WORLDARENA_REALCAM_I2V_DRY_RUN"):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "repo_root": str(repo_root),
                    "finetune_root": str(finetune_root),
                    "pretrained_model_path": str(pretrained_model),
                    "controlnet_checkpoint": str(controlnet_checkpoint),
                    "num_samples": len(rows),
                    "video_shape": [
                        int(generation.get("num_frames", generation.get("video_length", 81))),
                        int(generation.get("height", 512)),
                        int(generation.get("width", 896)),
                    ],
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
        finetune_root=finetune_root,
        pretrained_model=pretrained_model,
        controlnet_checkpoint=controlnet_checkpoint,
        generation=generation,
    )
    output_parents = {
        Path(str(row["output_path"])).expanduser().resolve().parent for row in rows
    }
    # Each GPU shard has its own UUID-bearing batch spec.  Keep the generated
    # model/camera metadata under that namespace so concurrent runners cannot
    # race on the same camera_poses.json file.
    work_root = next(iter(output_parents)) / "_realcam_i2v_work" / batch_spec_path.stem
    result_dir = work_root / "official_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_meta_path, camera_meta_path, camera_keys = _write_runtime_metadata(
        rows=rows,
        generation=generation,
        work_root=work_root,
        pretrained_model=pretrained_model,
        controlnet_checkpoint=controlnet_checkpoint,
    )

    with working_directory(finetune_root):
        runtime = _load_official_runtime(
            finetune_root=finetune_root,
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
