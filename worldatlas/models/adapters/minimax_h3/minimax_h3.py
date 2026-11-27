"""MiniMax H3 audio-video adapter for WorldAtlas.

MiniMax H3 (``MiniMax-AI/MiniMax-H3``) runs locally through SGLang's official
diffusion runtime (``sglang.multimodal_gen.DiffGenerator``). WorldAtlas drives
it in a persistent batch runner subprocess so the DiT, video VAE, audio VAE,
and Qwen3-VL encoder load once per benchmark shard.
"""

from __future__ import annotations

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class MiniMaxH3Adapter(ExternalBatchAdapter):
    """Generate joint audio+video with MiniMax H3 via SGLang DiffGenerator."""

    runner_module = "worldatlas.models.adapters.minimax_h3.minimax_h3_batch_runner"
    label = "MiniMax-H3"

    def batch_checkpoint_load_policy(self) -> str:
        """One SGLang DiffGenerator instance is loaded per shard and reused per row."""
        return "load_once"
