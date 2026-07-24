"""WorldCam model adapter for WorldAtlas Arena.

WorldCam is a camera-conditioned world model built on Wan2.1. This adapter
delegates inference to ``worldcam_runner`` as a subprocess so heavy GPU
dependencies stay isolated from the benchmark orchestration layer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from worldarena.benchmark.annotations import load_camera_matrices
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, run_command, run_sequential_generate_batch


WORLDCAM_TEMPORAL_STRIDE = 4
WORLDCAM_CONDITION_LATENTS = 8
WORLDCAM_INITIAL_OUTPUT_FRAMES = 1 + WORLDCAM_TEMPORAL_STRIDE * (WORLDCAM_CONDITION_LATENTS - 1)


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _resolve_worldcam_generation(
    sample: BenchmarkSample,
    generation: dict[str, Any],
) -> dict[str, int | float | None]:
    """Resolve sample-dependent ``auto`` and ``match_gt`` generation values."""
    poses = load_camera_matrices(sample.annotation_path)
    if poses is None or len(poses) == 0:
        raise FileNotFoundError(
            f"WorldCam requires non-empty poses.npy under annotation_path for {sample.sample_id}"
        )
    gt_frame_count = int(len(poses))

    raw_camera_frames = generation.get("camera_frames", "auto")
    camera_frames = gt_frame_count if _is_auto(raw_camera_frames) else int(raw_camera_frames)
    if camera_frames <= 0:
        raise ValueError(f"WorldCam camera_frames must be positive, got {camera_frames}")

    match_gt_frame_count = bool(generation.get("match_gt_frame_count", False))
    output_frames = gt_frame_count if match_gt_frame_count else None

    raw_num_ar_steps = generation.get("num_ar_steps", 50)
    if _is_auto(raw_num_ar_steps):
        if output_frames is None:
            raise ValueError("WorldCam num_ar_steps=auto requires match_gt_frame_count=true")
        generated_frames = max(output_frames - WORLDCAM_INITIAL_OUTPUT_FRAMES, 0)
        num_ar_steps = max(math.ceil(generated_frames / WORLDCAM_TEMPORAL_STRIDE), 1)
    else:
        num_ar_steps = int(raw_num_ar_steps)
    if num_ar_steps <= 0:
        raise ValueError(f"WorldCam num_ar_steps must be positive, got {num_ar_steps}")

    raw_output_fps = generation.get("output_fps", 30)
    match_gt_fps = bool(generation.get("match_gt_fps", False)) or (
        isinstance(raw_output_fps, str) and raw_output_fps.strip().lower() == "match_gt"
    )
    if match_gt_fps:
        if sample.fps is None or not math.isfinite(float(sample.fps)) or float(sample.fps) <= 0.0:
            raise ValueError(f"WorldCam match_gt_fps requires valid sample FPS for {sample.sample_id}")
        output_fps = float(sample.fps)
    else:
        output_fps = float(raw_output_fps)
    if not math.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError(f"WorldCam output_fps must be positive and finite, got {output_fps}")

    return {
        "camera_frames": camera_frames,
        "output_frames": output_frames,
        "output_fps": output_fps,
        "num_ar_steps": num_ar_steps,
        "gt_frame_count": gt_frame_count,
    }


def _ref_indices_arg(value: Any) -> str | None:
    """Normalize long-term memory ref indices from config into a CLI string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(item)) for item in value)
    return str(value)


class WorldCamAdapter(ModelAdapter):
    """Generate camera-guided videos from a conditioning clip and GT poses."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "reload_per_sample"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return run_sequential_generate_batch(self, requests)

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        # WorldCam conditions on extrinsics/intrinsics stored under annotation_path.
        if not sample.annotation_path:
            raise FileNotFoundError(
                f"WorldCam requires annotation_path with poses.npy/intrinsics.npy for {sample.sample_id}"
            )
        if self.config.checkpoint_dir is None:
            raise ValueError("WorldCam adapter requires checkpoint_dir")

        generation = self.config.generation
        resolved = _resolve_worldcam_generation(sample, generation)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        prompt_prefix = str(generation.get("prompt_prefix", ""))
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.worldcam_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--input_video",
            str(conditioning_image.expanduser().resolve()),
            "--annotation_path",
            str(Path(sample.annotation_path).expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--prompt",
            f"{prompt_prefix}{prompt}".strip(),
            "--height",
            str(int(generation.get("height", 480))),
            "--width",
            str(int(generation.get("width", 832))),
            "--cond_frames",
            str(int(generation.get("cond_frames", 65))),
            "--camera_frames",
            str(resolved["camera_frames"]),
            "--output_fps",
            str(resolved["output_fps"]),
            "--output_quality",
            str(int(generation.get("output_quality", 4))),
            "--cfg_scale",
            str(float(generation.get("cfg_scale", 4.0))),
            "--seed",
            str(int(generation.get("seed", 0))),
            "--num_ar_steps",
            str(resolved["num_ar_steps"]),
            "--long_term_memory_start",
            str(int(generation.get("long_term_memory_start", 30))),
            "--long_term_memory_num_clips",
            str(int(generation.get("long_term_memory_num_clips", 4))),
            "--device",
            str(generation.get("device", "cuda")),
            "--torch_dtype",
            str(generation.get("torch_dtype", "bfloat16")),
            "--weights_filename",
            str(generation.get("weights_filename", "finetuned_dit.safetensors")),
            "--negative_prompt",
            str(generation.get("negative_prompt", "")),
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        if resolved["output_frames"] is not None:
            command.extend(["--output_frames", str(resolved["output_frames"])])
        ref_indices = _ref_indices_arg(generation.get("long_term_memory_ref_indices"))
        if ref_indices:
            command.extend(["--long_term_memory_ref_indices", ref_indices])
        if generation.get("attention_sink_inference", False):
            command.append("--attention_sink_inference")

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"WorldCam output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": f"{prompt_prefix}{prompt}".strip(),
            "resolved_generation": resolved,
        }
