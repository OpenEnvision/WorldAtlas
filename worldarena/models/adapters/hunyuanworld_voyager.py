from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class HunyuanWorldVoyagerAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.hunyuanworld_voyager_batch_runner"
    label = "HunyuanWorld-Voyager"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        payload["camera_path"] = list(request.sample.camera_path)
        payload["prompt_current"] = request.sample.prompt_current
        payload["prompt_target"] = request.sample.prompt_target
        return payload

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        num_gpus = int(self.config.generation.get("num_gpus", 1))
        if num_gpus <= 1:
            return super()._build_command(spec_path, generation_path)
        if self.config.repo_root is None:
            raise ValueError(f"{self.label} adapter requires repo_root")
        if self.config.generation.get("use_cpu_offload", False):
            raise ValueError("HunyuanWorld-Voyager multi-GPU generation cannot use CPU offload")

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


__all__ = ["HunyuanWorldVoyagerAdapter"]
