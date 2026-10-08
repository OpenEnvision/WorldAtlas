"""LingBot-World V2 causal-fast pipeline behavior."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldatlas.models.adapters.lingbot_world.common import (
    LINGBOT_DEFAULT_FRAME_NUM,
    LingBotWorldBaseAdapter,
    _LingBotPersistentRuntime,
    resolve_frame_num,
)


class LingBotWorldV2Adapter(LingBotWorldBaseAdapter):
    """Generate camera-guided videos with WanI2VCausal."""

    def _pipeline_spec(self, wan: Any) -> tuple[Any, dict[str, Any]]:
        generation = self.config.generation
        return wan.WanI2VCausal, {
            "local_attn_size": int(generation.get("local_attn_size", 18)),
            "sink_size": int(generation.get("sink_size", 6)),
            "infer_mode": str(generation.get("infer_mode", "causal_fast")),
        }

    @staticmethod
    def _generation_kwargs_for_runtime(
        runtime: _LingBotPersistentRuntime,
        *,
        action_path: str | None,
        action_string: str | None,
        allow_act2cam: bool,
    ) -> dict[str, Any]:
        # v2 WanI2VCausal.generate has a causal-fast-only signature:
        #   (input_prompt, img, action_path, chunk_size, max_area, frame_num,
        #    timesteps_index, shift, seed, offload_model, max_sequence_length,
        #    max_attention_size)
        # It does NOT accept v1's vis_ui / sample_solver / sampling_steps /
        # guide_scale / allow_act2cam / action_string, and action_path is a required
        # positional (must be non-None; causal_fast asserts it and loads poses.npy).
        size = str(runtime.generation.get("size", "480*832"))
        if size not in runtime.max_area_configs:
            raise ValueError(f"unsupported LingBot size: {size}")
        if not action_path:
            raise ValueError(
                "lingbot-world-v2 requires a resolved action_path (poses.npy/intrinsics.npy dir)"
            )

        kwargs: dict[str, Any] = {
            "action_path": action_path,
            "chunk_size": int(runtime.generation.get("chunk_size", 3)),
            "max_area": runtime.max_area_configs[size],
            "frame_num": resolve_frame_num(
                runtime.generation,
                default=int(getattr(runtime.cfg, "frame_num", LINGBOT_DEFAULT_FRAME_NUM)),
            ),
            "shift": float(runtime.generation.get("sample_shift", runtime.cfg.sample_shift)),
            "seed": int(runtime.generation.get("base_seed", 42)),
            "offload_model": bool(runtime.offload_model),
            "max_sequence_length": int(runtime.generation.get("max_sequence_length", 512)),
        }
        timesteps_index = runtime.generation.get("timesteps_index")
        if timesteps_index is not None:
            kwargs["timesteps_index"] = [int(v) for v in timesteps_index]
        if runtime.generation.get("max_attention_size") is not None:
            kwargs["max_attention_size"] = int(runtime.generation["max_attention_size"])
        return kwargs

    @staticmethod
    def _resize_conditioning_image(image, target_width: int, target_height: int):
        """Resize-to-cover then center-crop to exactly (target_width, target_height).

        Replicates lingbot-video's preprocess_image (pipeline_lingbot_video_i2v.py)
        verbatim so v2 conditioning matches that official path: torch bilinear
        interpolation (align_corners=False, no antialias), math.ceil for the covered
        size, and round((new-target)/2) center-crop offsets. Keeps every conditioning
        image a fixed shape so the causal-fast latent grid (lat_h/lat_w derived from
        aspect_ratio) stays constant across samples. Returns a PIL image because the
        downstream pipeline re-tensorizes via TF.to_tensor.
        """
        import math

        import torch
        import torch.nn.functional as F
        from PIL import Image

        raw = (
            torch.from_numpy(np.asarray(image.convert("RGB")).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        old_h, old_w = raw.shape[-2:]
        scale = max(target_height / old_h, target_width / old_w)
        new_h = max(math.ceil(old_h * scale), target_height)
        new_w = max(math.ceil(old_w * scale), target_width)
        if (new_h, new_w) != (old_h, old_w):
            raw = F.interpolate(raw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = int(round((new_h - target_height) / 2.0))
        left = int(round((new_w - target_width) / 2.0))
        cropped = raw[:, :, top : top + target_height, left : left + target_width]
        arr = cropped.squeeze(0).permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()
        return Image.fromarray(arr)

