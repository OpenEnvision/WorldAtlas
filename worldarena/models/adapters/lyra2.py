from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command


class Lyra2Adapter(ModelAdapter):
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("Lyra-2 adapter requires repo_root")
        if not sample.annotation_path:
            raise FileNotFoundError(
                f"Lyra-2 requires annotation_path with poses.npy/intrinsics.npy for {sample.sample_id}"
            )

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.lyra2_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--annotation_path",
            str(Path(sample.annotation_path).expanduser().resolve()),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            sample.prediction_stem,
            "--prompt",
            prompt,
        ]
        if self.config.checkpoint_dir is not None:
            command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])

        extend_command_with_options(
            command,
            payload=generation,
            value_options={
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
        }
