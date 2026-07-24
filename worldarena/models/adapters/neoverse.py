"""NeoVerse model adapter for WorldAtlas Arena."""

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


class NeoVerseAdapter(ModelAdapter):
    """Generate videos via the NeoVerse upstream repo."""

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
            raise ValueError("NeoVerse adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("NeoVerse adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()
        reference_input = (
            conditioning_image
            if sample.conditioning_strategy == "reference_video" and conditioning_image.exists()
            else Path(sample.reference_path).expanduser().resolve()
        )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.neoverse_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--reference_path",
            str(reference_input),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            sample.prediction_stem,
            "--suite",
            sample.suite,
            "--prompt",
            prompt_text,
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        if sample.annotation_path:
            command.extend(
                [
                    "--annotation_path",
                    str(Path(sample.annotation_path).expanduser().resolve()),
                ]
            )

        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "negative_prompt": str,
                "input_mode": str,
                "trajectory_source": str,
                "trajectory": str,
                "traj_mode": str,
                "angle": float,
                "distance": float,
                "orbit_radius": float,
                "zoom_ratio": float,
                "num_frames": int,
                "height": int,
                "width": int,
                "resize_mode": str,
                "alpha_threshold": float,
                "seed": int,
                "model_subdir": str,
                "reconstructor_filename": str,
            },
            flag_options=("disable_lora", "low_vram", "vis_rendering"),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"NeoVerse output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
        }


__all__ = ["NeoVerseAdapter"]
