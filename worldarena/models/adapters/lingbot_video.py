"""LingBot-Video TI2V adapter for WorldArena.

The adapter batches requests into one subprocess so the upstream DiT and text
encoder are loaded once for an entire benchmark shard.  Both the Dense model
and the base stage of the MoE model use the same runner.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class LingBotVideoAdapter(ExternalBatchAdapter):
    """Generate first-frame-conditioned videos with LingBot-Video."""

    runner_module = "worldarena.models.adapters.lingbot_video_batch_runner"
    label = "LingBot-Video"

    def batch_checkpoint_load_policy(self) -> str:
        """The TI2V pipeline is loaded once before iterating over requests."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        payload.update(
            {
                "camera_path": list(request.sample.camera_path),
                "duration_seconds": request.sample.duration_seconds,
                "prompt_sequence": list(request.sample.prompt_sequence),
            }
        )
        return payload

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if self.config.repo_root is None:
            raise ValueError("LingBot-Video adapter requires repo_root")

        runner_args = [
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
            runner_args.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])

        nproc_per_node = int(self.config.generation.get("nproc_per_node", 1) or 1)
        if nproc_per_node <= 1:
            return [self.config.python_bin, *runner_args]

        command = [
            self.config.python_bin,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={nproc_per_node}",
        ]
        master_port = os.environ.get(
            "WORLDARENA_LOCAL_MASTER_PORT",
            self.config.generation.get("master_port"),
        )
        if master_port is not None:
            command.append(f"--master-port={int(master_port)}")
        command.extend(runner_args)
        return command
