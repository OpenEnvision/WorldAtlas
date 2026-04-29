from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command
from worldarena.models.adapters.pose_synthesis import resolve_pose_annotation_path


class Lyra2Adapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("Lyra-2 adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Lyra-2 adapter requires checkpoint_dir")

        generation = self.config.generation
        project_root = Path(__file__).resolve().parents[3]
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("Lyra-2 batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"lyra2_{uuid4().hex}.jsonl"
        pose_by_sample: dict[str, tuple[str, Path]] = {}
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                pose_annotation_path, pose_source = resolve_pose_annotation_path(
                    self.config,
                    request.sample,
                    request.conditioning_image.expanduser().resolve(),
                    frame_count=int(generation.get("num_frames", 81)),
                    namespace="lyra2",
                    focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
                )
                pose_by_sample[request.sample.sample_id] = (pose_source, pose_annotation_path)
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "prediction_stem": request.sample.prediction_stem,
                            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                            "output_path": str(request.output_path),
                            "prompt": request.prompt,
                            "annotation_path": str(pose_annotation_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.lyra2_batch_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "experiment": str,
                "height": int,
                "width": int,
                "num_frames": int,
                "guidance": float,
                "shift": float,
                "num_sampling_step": int,
                "seed": int,
                "fps": int,
                "pose_scale": float,
                "context_parallel_size": int,
                "prompt_suffix": str,
                "da3_model_name": str,
                "da3_model_path_custom": str,
                "da3_frame_interval": int,
                "da3_max_history_frames": int,
            },
            flag_options=(
                "offload",
                "offload_when_prompt",
                "debug",
                "da3_include_ar_chunk_last_frames",
                "da3_use_predicted_pose",
                "da3_predicted_pose_continuation",
                "ablate_same_t5",
                "use_dmd_scheduler",
                "disable_cache_update",
                "offload_da3_diffusion",
            ),
        )
        if "use_moge_scale" in generation:
            command.append("--use_moge_scale" if generation["use_moge_scale"] else "--no-use_moge_scale")

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            pose_source, pose_annotation_path = pose_by_sample[request.sample.sample_id]
            if request.output_path.exists():
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                    "pose_source": pose_source,
                    "pose_annotation_path": str(pose_annotation_path),
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"Lyra-2 output was not written: {request.output_path}",
                    "prompt": request.prompt,
                    "pose_source": pose_source,
                    "pose_annotation_path": str(pose_annotation_path),
                }
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("Lyra-2 adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Lyra-2 adapter requires checkpoint_dir")
        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        pose_annotation_path, pose_source = resolve_pose_annotation_path(
            self.config,
            sample,
            conditioning_image.expanduser().resolve(),
            frame_count=int(generation.get("num_frames", 81)),
            namespace="lyra2",
            focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
        )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.lyra2_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--annotation_path",
            str(pose_annotation_path),
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
                "experiment": str,
                "height": int,
                "width": int,
                "num_frames": int,
                "guidance": float,
                "shift": float,
                "num_sampling_step": int,
                "seed": int,
                "fps": int,
                "pose_scale": float,
                "context_parallel_size": int,
                "prompt_suffix": str,
                "da3_model_name": str,
                "da3_model_path_custom": str,
                "da3_frame_interval": int,
                "da3_max_history_frames": int,
            },
            flag_options=(
                "offload",
                "offload_when_prompt",
                "debug",
                "da3_include_ar_chunk_last_frames",
                "da3_use_predicted_pose",
                "da3_predicted_pose_continuation",
                "ablate_same_t5",
                "use_dmd_scheduler",
                "disable_cache_update",
                "offload_da3_diffusion",
            ),
        )
        if "use_moge_scale" in generation:
            command.append("--use_moge_scale" if generation["use_moge_scale"] else "--no-use_moge_scale")

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Lyra-2 output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
            "pose_source": pose_source,
            "pose_annotation_path": str(pose_annotation_path),
        }
