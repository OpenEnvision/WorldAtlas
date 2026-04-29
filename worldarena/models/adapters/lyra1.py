from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
    run_sequential_generate_batch,
)


def _view_indices_arg(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


class Lyra1Adapter(ModelAdapter):
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
        if self.config.repo_root is None:
            raise ValueError("Lyra-1 adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Lyra-1 adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.lyra1_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            sample.prediction_stem,
            "--prompt",
            prompt,
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])

        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "num_gpus": int,
                "num_video_frames": int,
                "height": int,
                "width": int,
                "fps": int,
                "seed": int,
                "guidance": float,
                "num_steps": int,
                "trajectory": str,
                "camera_rotation": str,
                "movement_distance": float,
                "total_movement_distance_factor": float,
                "target_index_subsample": int,
                "output_view_index": int,
                "negative_prompt": str,
            },
            flag_options=(
                "foreground_masking",
                "multi_trajectory",
                "offload_diffusion_transformer",
                "offload_tokenizer",
                "offload_text_encoder_model",
            ),
        )
        static_view_indices = _view_indices_arg(generation.get("static_view_indices_fixed"))
        if static_view_indices:
            command.extend(["--static_view_indices_fixed", static_view_indices])

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Lyra-1 output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }
