from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.adapters.common import build_env, run_command


class WorldFMAdapter(ModelAdapter):
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        del prompt
        if self.config.repo_root is None:
            raise ValueError("WorldFM adapter requires repo_root")
        if not sample.annotation_path:
            raise FileNotFoundError(
                f"WorldFM requires annotation_path with poses.npy/intrinsics.npy for {sample.sample_id}"
            )

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.worldfm_runner",
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
            "--model_filename",
            str(generation.get("model_filename", "worldfm_2-step.pth")),
            "--vae_subdir",
            str(generation.get("vae_subdir", "vae")),
            "--step",
            str(int(generation.get("step", 2))),
            "--image_size",
            str(int(generation.get("image_size", 512))),
            "--cfg_scale",
            str(float(generation.get("cfg_scale", 4.5))),
            "--render_size",
            str(int(generation.get("render_size", 512))),
            "--fps",
            str(int(generation.get("fps", 30))),
            "--gpu_index",
            str(int(generation.get("gpu_index", 0))),
        ]
        if self.config.checkpoint_dir is not None:
            command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        for option_name in ("config_path", "hw_path", "moge_path", "moge_pretrained"):
            option_value = generation.get(option_name)
            if option_value:
                command.extend([f"--{option_name}", str(option_value)])

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"WorldFM output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
        }
