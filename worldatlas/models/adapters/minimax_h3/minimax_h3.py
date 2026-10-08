"""MiniMax H3 audio-video adapter for WorldAtlas.

MiniMax H3 was ported into the sibling **WorldFoundry** project as
``worldfoundry.pipelines.minimax.pipeline_minimax_h3.NativeMiniMaxH3Pipeline``.
WorldAtlas drives it out of process (like the other heavy video models) so the
bespoke DiT / video VAE / audio VAE / Qwen3-VL encoder load once per benchmark
shard.  The subprocess must run under an interpreter that can import *both*
``worldatlas`` (for the batch-runner helpers) and ``worldfoundry`` (for the
pipeline); point ``python_bin`` at that unified environment and ``repo_root`` at
the WorldFoundry checkout in the model YAML.
"""

from __future__ import annotations

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class MiniMaxH3Adapter(ExternalBatchAdapter):
    """Generate joint audio+video with WorldFoundry's NativeMiniMaxH3Pipeline."""

    runner_module = "worldatlas.models.adapters.minimax_h3.minimax_h3_batch_runner"
    label = "MiniMax-H3"

    def batch_checkpoint_load_policy(self) -> str:
        """The H3 pipeline is built once via from_pretrained, then reused per row."""
        return "load_once"
