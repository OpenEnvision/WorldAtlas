from __future__ import annotations

from pathlib import Path

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


_STATIC_CAMERA_ACTIONS = {
    "orbit_left": "a leftward orbit around the scene",
    "orbit_right": "a rightward orbit around the scene",
    "pan_left": "a left pan",
    "pan_right": "a right pan",
    "push_in": "a forward push-in toward the scene",
    "pull_out": "a backward pull-out away from the scene",
    "move_left": "a leftward sideways truck",
    "move_right": "a rightward sideways truck",
}


def _clean_static_scene_prompt(prompt: str) -> str:
    sentences = [part.strip() for part in prompt.replace("\n", " ").split(".")]
    kept: list[str] = []
    for sentence in sentences:
        if not sentence:
            continue
        lowered = sentence.lower()
        if lowered.startswith("camera "):
            continue
        if "everything" in lowered and "motionless" in lowered:
            continue
        kept.append(sentence)
    return ". ".join(kept).strip()


class HunyuanVideoAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.hunyuan_video_i2v_batch_runner"
    label = "HunyuanVideo-I2V"

    def prompt_for(self, sample: BenchmarkSample) -> str:
        prompt = super().prompt_for(sample)
        if (
            sample.suite != "image_static"
            or not self.config.generation.get("rewrite_static_camera_prompt", False)
        ):
            return prompt

        actions = [
            _STATIC_CAMERA_ACTIONS.get(action, action.replace("_", " "))
            for action in sample.camera_path
            if action != "fixed"
        ]
        if not actions:
            return prompt

        if len(actions) == 1:
            camera_text = f"The camera makes {actions[0]} with clear, continuous motion and visible parallax."
        else:
            camera_text = (
                "The camera changes viewpoint with visible parallax, moving through this sequence: "
                + ", then ".join(actions)
                + "."
            )
        scene_text = _clean_static_scene_prompt(prompt)
        if scene_text and not scene_text.endswith("."):
            scene_text = f"{scene_text}."
        parts = [
            camera_text,
            scene_text,
        ]
        return " ".join(part for part in parts if part).strip()

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        num_gpus = int(self.config.generation.get("num_gpus", 1))
        if num_gpus <= 1:
            return super()._build_command(spec_path, generation_path)
        if self.config.repo_root is None:
            raise ValueError(f"{self.label} adapter requires repo_root")
        if self.config.generation.get("use_cpu_offload", False):
            raise ValueError("HunyuanVideo-I2V multi-GPU generation cannot use CPU offload")

        torchrun_bin = str(Path(self.config.python_bin).with_name("torchrun"))
        command = [
            torchrun_bin,
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(num_gpus),
            "-m",
            self.runner_module,
            "--repo_root",
            str(self.config.repo_root),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ]
        if self.config.checkpoint_dir is not None:
            command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        return command
