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


def _generation_for_suite(generation: dict[str, Any], suite: str) -> dict[str, Any]:
    resolved = dict(generation)
    for key, value in generation.items():
        if not key.endswith("_by_suite") or not isinstance(value, dict):
            continue
        base_key = key[: -len("_by_suite")]
        if suite in value:
            resolved[base_key] = value[suite]
    return resolved


class LucidDreamerAdapter(ModelAdapter):
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
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("LucidDreamer adapter requires repo_root")

        generation = _generation_for_suite(self.config.generation, sample.suite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.luciddreamer_runner",
            "--repo_root",
            str(repo_root),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt_text,
            "--campath_gen",
            str(generation.get("campath_gen", "lookdown")),
            "--campath_render",
            str(generation.get("campath_render", "llff")),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "negative_prompt": str,
                "model_name": str,
                "seed": int,
                "diff_steps": int,
            },
            bool_value_options=("keep_work_dir",),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"LucidDreamer output was not written: {output_path}")

        ply_path = output_path.with_suffix(".ply")
        depth_video_path = output_path.with_name(f"{output_path.stem}.depth{output_path.suffix}")
        payload: dict[str, Any] = {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "campath_gen": str(generation.get("campath_gen", "lookdown")),
            "campath_render": str(generation.get("campath_render", "llff")),
        }
        if ply_path.exists():
            payload["gaussian_splat_path"] = str(ply_path)
        if depth_video_path.exists():
            payload["depth_video_path"] = str(depth_video_path)
        return payload


__all__ = ["LucidDreamerAdapter", "_generation_for_suite"]
