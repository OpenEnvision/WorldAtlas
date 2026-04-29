from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, run_command, run_sequential_generate_batch


def _ref_indices_arg(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(item)) for item in value)
    return str(value)


class WorldCamAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

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
        if not sample.annotation_path:
            raise FileNotFoundError(
                f"WorldCam requires annotation_path with poses.npy/intrinsics.npy for {sample.sample_id}"
            )
        if self.config.checkpoint_dir is None:
            raise ValueError("WorldCam adapter requires checkpoint_dir")

        generation = self.config.generation
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
            str(int(generation.get("camera_frames", 233))),
            "--output_fps",
            str(int(generation.get("output_fps", 30))),
            "--output_quality",
            str(int(generation.get("output_quality", 4))),
            "--cfg_scale",
            str(float(generation.get("cfg_scale", 4.0))),
            "--seed",
            str(int(generation.get("seed", 0))),
            "--num_ar_steps",
            str(int(generation.get("num_ar_steps", 50))),
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
        }
