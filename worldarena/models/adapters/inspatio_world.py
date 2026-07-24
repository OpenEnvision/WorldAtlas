"""InSpatio-World model adapter for WorldAtlas Arena."""

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


def _gpu_list_arg(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
        return token or None
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


class InSpatioWorldAdapter(ModelAdapter):
    """Generate spatially consistent videos via the InSpatio-World upstream repo."""

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
        if self.config.repo_root is None:
            raise ValueError("InSpatio-World adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("InSpatio-World adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.inspatio_world_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--input_video",
            str(conditioning_image.expanduser().resolve()),
            "--prompt",
            prompt,
            "--output_path",
            str(output_path),
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "traj_txt_path": str,
                "config_path": str,
                "checkpoint_path": str,
                "da3_repo_root": str,
                "da3_model_path": str,
                "tae_checkpoint_path": str,
                "step2_gpus": _gpu_list_arg,
                "step3_gpus": _gpu_list_arg,
                "step3_nproc": int,
                "master_port": int,
                "freeze_repeat": int,
                "freeze_frame": int,
                "radius_ratio": float,
            },
            bool_value_options=[
                "relative_to_source",
                "rotation_only",
                "adaptive_frame",
                "use_tae",
                "compile_dit",
                "keep_work_dir",
            ],
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"InSpatio-World output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }


__all__ = ["InSpatioWorldAdapter"]
