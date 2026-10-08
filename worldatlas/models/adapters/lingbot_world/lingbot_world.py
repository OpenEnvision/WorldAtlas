"""LingBot-World V1 base and fast pipeline behavior."""

from __future__ import annotations

from typing import Any

from worldatlas.models.adapters.lingbot_world.common import (
    LINGBOT_DEFAULT_FRAME_NUM,
    LingBotWorldBaseAdapter,
    _LingBotPersistentRuntime,
    _is_fast_entrypoint,
    resolve_frame_num,
)


class LingBotWorldAdapter(LingBotWorldBaseAdapter):
    """Generate camera-guided videos with WanI2V or WanI2VFast."""

    def _pipeline_spec(self, wan: Any) -> tuple[Any, dict[str, Any]]:
        if not _is_fast_entrypoint(self.config):
            return wan.WanI2V, {}
        # Leave upstream defaults intact unless the attention window is configured.
        options = {
            key: int(self.config.generation[key])
            for key in ("local_attn_size", "sink_size")
            if self.config.generation.get(key) is not None
        }
        return wan.WanI2VFast, options

    @staticmethod
    def _generation_kwargs_for_runtime(
        runtime: _LingBotPersistentRuntime,
        *,
        action_path: str | None,
        action_string: str | None,
        allow_act2cam: bool,
    ) -> dict[str, Any]:
        size = str(runtime.generation.get("size", "832*480"))
        if size not in runtime.max_area_configs:
            raise ValueError(f"unsupported LingBot size: {size}")

        kwargs: dict[str, Any] = {
            "action_path": action_path,
            "max_area": runtime.max_area_configs[size],
            "frame_num": resolve_frame_num(
                runtime.generation,
                default=int(getattr(runtime.cfg, "frame_num", LINGBOT_DEFAULT_FRAME_NUM)),
            ),
            "shift": float(runtime.generation.get("sample_shift", runtime.cfg.sample_shift)),
            "seed": int(runtime.generation.get("base_seed", 42)),
            "offload_model": bool(runtime.offload_model),
        }
        if runtime.entrypoint_name == "generate_fast.py":
            if action_string:
                raise ValueError("generate_fast.py does not support action_string or allow_act2cam")
            if allow_act2cam and action_path:
                raise ValueError("generate_fast.py does not support allow_act2cam")
            kwargs["chunk_size"] = int(runtime.generation.get("chunk_size", 3))
            if runtime.generation.get("max_attention_size") is not None:
                kwargs["max_attention_size"] = int(runtime.generation["max_attention_size"])
            # WanI2VFast.generate() has no vis_ui; the full WanI2V path does.
            return kwargs

        guide_scale = runtime.generation.get("sample_guide_scale")
        if guide_scale is None:
            guide_scale = runtime.cfg.sample_guide_scale
        elif isinstance(guide_scale, list):
            guide_scale = tuple(float(value) for value in guide_scale)
        elif isinstance(guide_scale, tuple):
            guide_scale = tuple(float(value) for value in guide_scale)
        else:
            guide_scale = float(guide_scale)

        kwargs["vis_ui"] = bool(runtime.generation.get("vis_ui", False))
        kwargs["allow_act2cam"] = bool(allow_act2cam)
        kwargs["action_string"] = action_string
        kwargs["sample_solver"] = str(runtime.generation.get("sample_solver", "unipc"))
        kwargs["sampling_steps"] = int(runtime.generation.get("sample_steps", runtime.cfg.sample_steps))
        kwargs["guide_scale"] = guide_scale
        return kwargs

    @staticmethod
    def _resize_conditioning_image(image: Any, width: int, height: int) -> Any:
        from PIL import Image

        return image.resize((width, height), Image.Resampling.LANCZOS)
