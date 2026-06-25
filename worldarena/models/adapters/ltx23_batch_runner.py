from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import traceback
from typing import Any
from contextlib import contextmanager

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena LTX-2.3 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _add_ltx_repo_to_sys_path(repo_root: Path) -> None:
    candidates = [
        repo_root / "packages" / "ltx-pipelines" / "src",
        repo_root / "packages" / "ltx-core" / "src",
        repo_root / "packages" / "ltx-trainer" / "src",
        repo_root,
    ]
    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _resolve_path(value: str | None, *, base_dir: Path, default: str | None = None) -> Path:
    raw = value if value not in (None, "") else default
    if raw is None:
        raise ValueError("missing required LTX-2.3 path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _install_persistent_weight_cache(enabled: bool) -> None:
    """Keep LTX modules alive for the whole runner process.

    The upstream LTX pipeline intentionally builds each block on every call and
    then moves it to ``meta`` to free memory. WorldArena shards call the same
    pipeline repeatedly, so that behavior would reread checkpoints for every
    sample. This runner-level patch changes the lifecycle to:

    1. build each unique SingleGPUModelBuilder once from checkpoint into CPU RAM;
    2. move the cached module to the active CUDA device for use;
    3. move it back to CPU after the block finishes, preserving weights.

    It is process-local, affects only this batch runner, and can be disabled by
    setting generation.persistent_weight_cache=false.
    """
    if not enabled:
        return

    import torch
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    import ltx_pipelines.utils.blocks as blocks_module
    import ltx_pipelines.utils.gpu_model as gpu_model_module
    from ltx_pipelines.utils.helpers import cleanup_memory

    if getattr(SingleGPUModelBuilder, "_worldarena_persistent_cache_installed", False):
        return

    original_build = SingleGPUModelBuilder.build
    cache: dict[tuple[Any, ...], Any] = {}

    def _path_key(value: Any) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            values = value
        else:
            values = (value,)
        return tuple(str(Path(str(item)).expanduser().resolve(strict=False)) for item in values)

    def _op_key(value: Any) -> tuple[Any, ...] | None:
        if value is None:
            return None
        return (
            value.__class__.__module__,
            value.__class__.__qualname__,
            getattr(value, "name", None),
        )

    def _builder_key(builder: Any, dtype: torch.dtype | None) -> tuple[Any, ...]:
        return (
            builder.model_class_configurator,
            _path_key(builder.model_path),
            _op_key(builder.model_sd_ops),
            tuple(_op_key(item) for item in builder.module_ops),
            tuple(
                (
                    str(Path(str(lora.path)).expanduser().resolve(strict=False)),
                    float(lora.strength),
                    _op_key(lora.sd_ops),
                )
                for lora in builder.loras
            ),
            type(builder.model_loader),
            (
                getattr(builder.fuse_rule, "aggregation_dtype", None),
                getattr(getattr(builder.fuse_rule, "fuse_fn", None), "__module__", None),
                getattr(getattr(builder.fuse_rule, "fuse_fn", None), "__qualname__", None),
            ),
            dtype,
        )

    def cached_build(
        self: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> Any:
        requested_device = torch.device("cuda") if device is None else torch.device(device)
        key = _builder_key(self, dtype)
        model = cache.get(key)
        if model is None:
            print_status(
                "ltx23-weight-cache",
                "loading",
                model_path=",".join(_path_key(self.model_path)),
                configurator=getattr(self.model_class_configurator, "__name__", str(self.model_class_configurator)),
            )
            model = original_build(self, device=torch.device("cpu"), dtype=dtype, **kwargs)
            model.eval()
            cache[key] = model
        return model.to(requested_device).eval()

    @contextmanager
    def cached_gpu_model(model: Any):
        try:
            yield model
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            model.to("cpu")
            cleanup_memory()

    SingleGPUModelBuilder.build = cached_build  # type: ignore[method-assign]
    SingleGPUModelBuilder._worldarena_persistent_cache_installed = True  # type: ignore[attr-defined]
    SingleGPUModelBuilder._worldarena_original_build = original_build  # type: ignore[attr-defined]
    gpu_model_module.gpu_model = cached_gpu_model
    blocks_module.gpu_model = cached_gpu_model


def _install_persistent_streaming_cache(
    enabled: bool,
    *,
    gpu_slots: int | None,
    keep_gpu: bool,
) -> None:
    """Reuse LTX streaming weights while keeping decoder headroom.

    ``keep_gpu=false`` caches streaming block weights and non-block weights in
    CPU memory, then rebuilds only the small GPU-side wrapper/pool for each
    stage. That avoids rereading checkpoints per sample without pinning the
    transformer on GPU between stages.
    """
    if not enabled and (gpu_slots is None or gpu_slots <= 0):
        return

    import atexit
    import torch
    from ltx_core.block_streaming.pool import WeightPool
    from ltx_core.block_streaming.provider import WeightsProvider
    from ltx_core.block_streaming.utils import resolve_attr
    from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
    import ltx_pipelines.utils.blocks as blocks_module
    from ltx_pipelines.utils.helpers import cleanup_memory
    from ltx_pipelines.utils.types import OffloadMode

    if getattr(blocks_module, "_worldarena_streaming_cache_installed", False):
        return

    cache: dict[tuple[Any, ...], Any] = {}
    gpu_slots_count = gpu_slots if gpu_slots is not None and gpu_slots > 0 else None

    def _path_key(value: Any) -> tuple[str, ...]:
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(str(Path(str(item)).expanduser().resolve(strict=False)) for item in values)

    def _op_key(value: Any) -> tuple[Any, ...] | None:
        if value is None:
            return None
        return (
            value.__class__.__module__,
            value.__class__.__qualname__,
            getattr(value, "name", None),
        )

    def _builder_key(
        builder: Any,
        *,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int | None,
    ) -> tuple[Any, ...]:
        return (
            builder.model_class_configurator,
            _path_key(builder.model_path),
            _op_key(builder.model_sd_ops),
            tuple(_op_key(item) for item in builder.module_ops),
            tuple(
                (
                    str(Path(str(lora.path)).expanduser().resolve(strict=False)),
                    float(lora.strength),
                    _op_key(lora.sd_ops),
                )
                for lora in builder.loras
            ),
            type(builder.model_loader),
            (
                getattr(builder.fuse_rule, "aggregation_dtype", None),
                getattr(getattr(builder.fuse_rule, "fuse_fn", None), "__module__", None),
                getattr(getattr(builder.fuse_rule, "fuse_fn", None), "__qualname__", None),
            ),
            builder.blocks_attr,
            builder.blocks_prefix,
            str(target_device),
            dtype,
            cpu_slots_count,
            gpu_slots_count,
        )

    def _copy_non_meta_state_to_cpu(model: Any) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if getattr(tensor, "device", None) is not None and tensor.device.type != "meta":
                state[name] = tensor.detach().to(device=torch.device("cpu"), copy=True)
        return state

    def _release_wrapper_gpu_only(wrapped: Any) -> None:
        provider = getattr(wrapped, "_provider", None)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        if provider is not None:
            try:
                provider._copy_stream.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.current_stream(provider._target_device).synchronize()
            except Exception:
                pass
        for handle in list(getattr(wrapped, "_hooks", [])):
            try:
                handle.remove()
            except Exception:
                pass
        try:
            wrapped._hooks.clear()
        except Exception:
            pass
        try:
            wrapped.to_empty(device=torch.device("meta"))
        except Exception:
            try:
                wrapped.to("meta")
            except Exception:
                pass
        if provider is not None:
            try:
                provider._cache.clear()
                provider._events.clear()
                provider._pool = None
            except Exception:
                pass
        cleanup_memory()

    def _cpu_cache_entry(builder: Any, wrapped: Any) -> dict[str, Any]:
        provider = wrapped._provider
        return {
            "config": builder.model_config(),
            "source": provider._source,
            "lora_sources": provider._lora_sources,
            "non_block_state": _copy_non_meta_state_to_cpu(wrapped._model),
        }

    def _build_from_cpu_cache(
        builder: Any,
        entry: dict[str, Any],
        *,
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        model = builder.meta_model(entry["config"], builder.module_ops)
        model.eval()
        non_block_state = {
            name: tensor.to(device=target_device, dtype=dtype)
            for name, tensor in entry["non_block_state"].items()
        }
        model.load_state_dict(non_block_state, strict=False, assign=True)
        blocks = resolve_attr(model, builder.blocks_attr)
        copy_stream = torch.cuda.Stream(device=target_device)
        gpu_pool = WeightPool(
            entry["source"].block_layout,
            gpu_slots_count or 2,
            target_device,
            reuse_barrier=lambda event: copy_stream.wait_event(event),
        )
        provider = WeightsProvider(
            gpu_pool,
            copy_stream,
            target_device,
            entry["source"],
            entry["lora_sources"],
            builder.blocks_prefix,
            fuse_rule=builder.fuse_rule,
        )
        return BlockStreamingWrapper(
            model=model,
            blocks=blocks,
            provider=provider,
            target_device=target_device,
        )

    @contextmanager
    def cached_streaming_model(
        builder: Any,
        offload_mode: Any,
        target_device: torch.device,
        dtype: torch.dtype,
    ):
        configurator_name = getattr(builder.model_class_configurator, "__name__", str(builder.model_class_configurator))
        persistent_gpu = enabled and keep_gpu and configurator_name == "LTXModelConfigurator"
        persistent_cpu = enabled and not persistent_gpu
        cpu_slots_count = blocks_module.DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None
        key = _builder_key(
            builder,
            target_device=target_device,
            dtype=dtype,
            cpu_slots_count=cpu_slots_count,
        )
        cached = cache.get(key) if enabled else None
        if persistent_cpu and cached is not None:
            wrapped = _build_from_cpu_cache(
                builder,
                cached,
                target_device=target_device,
                dtype=dtype,
            )
        else:
            wrapped = cached if persistent_gpu else None
        if wrapped is None:
            print_status(
                "ltx23-streaming-cache",
                "loading",
                model_path=",".join(_path_key(builder.model_path)),
                configurator=configurator_name,
                gpu_slots=gpu_slots_count,
                persistent_gpu=persistent_gpu,
                offload_mode=str(offload_mode),
            )
            wrapped = builder.build(
                target_device=target_device,
                dtype=dtype,
                cpu_slots_count=cpu_slots_count,
                gpu_slots_count=gpu_slots_count,
            )
            if persistent_gpu:
                cache[key] = wrapped
            elif persistent_cpu:
                cache[key] = _cpu_cache_entry(builder, wrapped)
        try:
            yield wrapped
        finally:
            if persistent_cpu:
                _release_wrapper_gpu_only(wrapped)
            elif not persistent_gpu:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                try:
                    wrapped.teardown()
                    wrapped.to("meta")
                except Exception:
                    pass
            cleanup_memory()

    def cleanup_cached_streaming_models() -> None:
        for cached in cache.values():
            if isinstance(cached, dict):
                try:
                    cached["source"].cleanup()
                except Exception:
                    pass
                for lora_source in cached.get("lora_sources", []):
                    try:
                        lora_source.cleanup()
                    except Exception:
                        pass
                try:
                    cached["non_block_state"].clear()
                except Exception:
                    pass
            else:
                try:
                    cached.teardown()
                    cached.to("meta")
                except Exception:
                    pass
        cache.clear()

    blocks_module._streaming_model = cached_streaming_model
    blocks_module._worldarena_streaming_cache_installed = True
    blocks_module._worldarena_streaming_cache = cache
    if enabled:
        atexit.register(cleanup_cached_streaming_models)


def _resolve_existing_path(
    value: str | None,
    *,
    base_dir: Path,
    default: str | None = None,
    label: str,
) -> str:
    path = _resolve_path(value, base_dir=base_dir, default=default)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _snap_frame_count(value: float) -> int:
    minimum = max(1, int(value + 0.999999))
    return ((minimum - 1 + 7) // 8) * 8 + 1


def _frame_count(generation: dict[str, Any]) -> int:
    fps = float(generation.get("frame_rate", generation.get("fps", 24.0)))
    requested = int(generation.get("num_frames", 121))
    min_seconds = float(generation.get("min_seconds", 5.0))
    return max(_snap_frame_count(requested), _snap_frame_count(fps * min_seconds))


def _seed_for_row(generation: dict[str, Any], row_index: int) -> int:
    seed = int(generation.get("seed", 10))
    if bool(generation.get("increment_seed", True)):
        seed += row_index
    return seed


def _prompt_context_to_device(context: Any, device: Any) -> Any:
    return context._replace(
        video_encoding=context.video_encoding.to(device),
        audio_encoding=context.audio_encoding.to(device) if context.audio_encoding is not None else None,
        attention_mask=context.attention_mask.to(device),
    )


def _prompt_context_to_cpu(context: Any) -> Any:
    return context._replace(
        video_encoding=context.video_encoding.detach().to("cpu"),
        audio_encoding=context.audio_encoding.detach().to("cpu") if context.audio_encoding is not None else None,
        attention_mask=context.attention_mask.detach().to("cpu"),
    )


def _preencode_prompt_contexts(
    *,
    pipeline: Any,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
) -> dict[int, Any]:
    if not _as_bool(generation.get("preencode_prompts"), default=False):
        return {}
    if _as_bool(generation.get("enhance_prompt"), default=False):
        raise ValueError("preencode_prompts requires enhance_prompt=false")

    import torch
    from ltx_pipelines.utils.gpu_model import gpu_model

    prompt_encoder = pipeline.prompt_encoder
    contexts: dict[int, Any] = {}
    print_status("ltx23-prompt-cache", "preencoding", records=len(rows))
    with (
        prompt_encoder._text_encoder_ctx() as text_encoder,
        gpu_model(prompt_encoder._build_embeddings_processor()) as embeddings_processor,
    ):
        for row_index, row in enumerate(rows):
            with torch.inference_mode():
                hidden_states, attention_mask = text_encoder.encode(str(row["prompt"]))
                context = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
            contexts[row_index] = _prompt_context_to_cpu(context)
            del hidden_states, attention_mask, context
            if torch.cuda.is_available() and (row_index + 1) % 8 == 0:
                torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_status("ltx23-prompt-cache", "preencoded", records=len(contexts))
    return contexts


def _ltx_lora(path: Path, strength: float):
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    return LoraPathStrengthAndSDOps(str(path), float(strength), LTXV_LORA_COMFY_RENAMING_MAP)


def _parse_lora_item(item: Any, *, base_dir: Path) -> Any:
    if isinstance(item, str):
        return _ltx_lora(_resolve_path(item, base_dir=base_dir), 1.0)
    if isinstance(item, dict):
        return _ltx_lora(
            _resolve_path(str(item["path"]), base_dir=base_dir),
            float(item.get("strength", 1.0)),
        )
    if isinstance(item, (list, tuple)) and item:
        return _ltx_lora(_resolve_path(str(item[0]), base_dir=base_dir), float(item[1] if len(item) > 1 else 1.0))
    raise ValueError(f"invalid LTX-2.3 LoRA spec: {item!r}")


def _parse_loras(value: Any, *, base_dir: Path) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, (str, dict)):
        return [_parse_lora_item(value, base_dir=base_dir)]
    return [_parse_lora_item(item, base_dir=base_dir) for item in value]


def _parse_quantization(value: str | None, *, checkpoint_path: str) -> Any:
    if not value:
        return None
    from ltx_pipelines.utils.quantization_factory import QuantizationKind

    return QuantizationKind(str(value)).to_policy(checkpoint_path=checkpoint_path)


def _guider_params(prefix: str, generation: dict[str, Any], default_params: Any) -> Any:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(
        cfg_scale=float(generation.get(f"{prefix}_cfg_guidance_scale", default_params.cfg_scale)),
        stg_scale=float(generation.get(f"{prefix}_stg_guidance_scale", default_params.stg_scale)),
        rescale_scale=float(generation.get(f"{prefix}_rescale_scale", default_params.rescale_scale)),
        modality_scale=float(generation.get(f"{prefix}_modality_guidance_scale", default_params.modality_scale)),
        skip_step=int(generation.get(f"{prefix}_skip_step", default_params.skip_step)),
        stg_blocks=[int(value) for value in generation.get(f"{prefix}_stg_blocks", default_params.stg_blocks)],
    )


def _generate_distilled_video_only(
    *,
    pipeline: Any,
    prompt: str,
    prompt_context: Any | None,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    images: list[Any],
    tiling_config: Any,
    enhance_prompt: bool,
    single_stage: bool,
    single_stage_full_resolution: bool,
) -> tuple[Any, None]:
    """Run the distilled two-stage pipeline without audio generation."""
    import torch
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
    from ltx_pipelines.utils.types import ModalitySpec

    assert_resolution(height=height, width=width, is_two_stage=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    dtype = torch.bfloat16

    if prompt_context is None:
        (ctx_p,) = pipeline.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
    else:
        ctx_p = _prompt_context_to_device(prompt_context, pipeline.device)
    video_context = ctx_p.video_encoding

    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    if single_stage and single_stage_full_resolution:
        stage_1_w, stage_1_h = width, height
    else:
        stage_1_w, stage_1_h = width // 2, height // 2
    stage_1_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=stage_1_h,
            width=stage_1_w,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )
    video_state, _audio_state = pipeline.stage(
        denoiser=SimpleDenoiser(video_context, None),
        sigmas=stage_1_sigmas,
        noiser=noiser,
        width=stage_1_w,
        height=stage_1_h,
        frames=num_frames,
        fps=frame_rate,
        video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
        audio=None,
    )
    if video_state is None:
        raise RuntimeError("LTX-2.3 video-only stage 1 returned no video state")
    if single_stage:
        del stage_1_conditionings, ctx_p, video_context
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        decoded_video = pipeline.video_decoder(video_state.latent, tiling_config, generator)
        return decoded_video, None

    upscaled_video_latent = pipeline.upsampler(video_state.latent[:1])
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    stage_2_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )
    video_state, _audio_state = pipeline.stage(
        denoiser=SimpleDenoiser(video_context, None),
        sigmas=stage_2_sigmas,
        noiser=noiser,
        width=width,
        height=height,
        frames=num_frames,
        fps=frame_rate,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_2_conditionings,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=None,
    )
    if video_state is None:
        raise RuntimeError("LTX-2.3 video-only stage 2 returned no video state")

    decoded_video = pipeline.video_decoder(video_state.latent, tiling_config, generator)
    return decoded_video, None


def _common_runtime(generation: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_base = checkpoint_dir
    distilled_checkpoint_path = _resolve_existing_path(
        generation.get("distilled_checkpoint_path"),
        base_dir=checkpoint_base,
        default="ltx-2.3-22b-distilled-1.1.safetensors",
        label="LTX-2.3 distilled checkpoint",
    )
    checkpoint_path = _resolve_path(
        generation.get("checkpoint_path"),
        base_dir=checkpoint_base,
        default="ltx-2.3-22b-dev.safetensors",
    )
    spatial_upsampler_path = _resolve_existing_path(
        generation.get("spatial_upsampler_path"),
        base_dir=checkpoint_base,
        default="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        label="LTX-2.3 spatial upsampler",
    )
    gemma_root = _resolve_existing_path(
        generation.get("gemma_root"),
        base_dir=checkpoint_base,
        default="../gemma-3-12b-it-qat-q4_0-unquantized",
        label="LTX-2.3 Gemma root",
    )
    return {
        "checkpoint_base": checkpoint_base,
        "checkpoint_path": str(checkpoint_path),
        "distilled_checkpoint_path": distilled_checkpoint_path,
        "spatial_upsampler_path": spatial_upsampler_path,
        "gemma_root": gemma_root,
        "height": int(generation.get("height", 512)),
        "width": int(generation.get("width", 768)),
        "num_frames": _frame_count(generation),
        "frame_rate": float(generation.get("frame_rate", generation.get("fps", 24.0))),
        "image_strength": float(generation.get("image_conditioning_strength", 1.0)),
        "image_crf": int(generation.get("image_crf", 33)),
        "enhance_prompt": bool(generation.get("enhance_prompt", False)),
    }


def _load_pipeline(generation: dict[str, Any], runtime: dict[str, Any]) -> Any:
    from ltx_pipelines.utils.types import OffloadMode

    pipeline_name = str(generation.get("pipeline", "distilled")).lower()
    offload_mode = OffloadMode(str(generation.get("offload", "none")))
    quantization = _parse_quantization(generation.get("quantization"), checkpoint_path=runtime["distilled_checkpoint_path"])

    if pipeline_name == "distilled":
        from ltx_pipelines.distilled import DistilledPipeline

        return DistilledPipeline(
            distilled_checkpoint_path=runtime["distilled_checkpoint_path"],
            gemma_root=runtime["gemma_root"],
            spatial_upsampler_path=runtime["spatial_upsampler_path"],
            loras=_parse_loras(generation.get("lora"), base_dir=runtime["checkpoint_base"]),
            quantization=quantization,
            offload_mode=offload_mode,
        )

    checkpoint_path = runtime["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"LTX-2.3 checkpoint not found: {checkpoint_path}")
    quantization = _parse_quantization(generation.get("quantization"), checkpoint_path=checkpoint_path)
    distilled_lora = _parse_loras(
        generation.get(
            "distilled_lora",
            [["ltx-2.3-22b-distilled-lora-384-1.1.safetensors", 0.8]],
        ),
        base_dir=runtime["checkpoint_base"],
    )
    loras = _parse_loras(generation.get("lora"), base_dir=runtime["checkpoint_base"])

    if pipeline_name in {"two_stage", "ti2vid_two_stages"}:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        return TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=runtime["spatial_upsampler_path"],
            gemma_root=runtime["gemma_root"],
            loras=loras,
            quantization=quantization,
            offload_mode=offload_mode,
        )

    if pipeline_name in {"two_stage_hq", "hq", "ti2vid_two_stages_hq"}:
        from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

        return TI2VidTwoStagesHQPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            distilled_lora_strength_stage_1=float(generation.get("distilled_lora_strength_stage_1", 0.25)),
            distilled_lora_strength_stage_2=float(generation.get("distilled_lora_strength_stage_2", 0.5)),
            spatial_upsampler_path=runtime["spatial_upsampler_path"],
            gemma_root=runtime["gemma_root"],
            loras=tuple(loras),
            quantization=quantization,
            offload_mode=offload_mode,
        )

    raise ValueError(f"unsupported LTX-2.3 pipeline: {pipeline_name}")


def _generate_one(
    *,
    pipeline: Any,
    row: dict[str, Any],
    row_index: int,
    generation: dict[str, Any],
    runtime: dict[str, Any],
    prompt_contexts: dict[int, Any] | None = None,
) -> Path:
    import torch
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, LTX_2_3_PARAMS
    from ltx_pipelines.utils.media_io import encode_video

    output_path = Path(str(row["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
    images = [
        ImageConditioningInput(
            path=str(image_path),
            frame_idx=0,
            strength=runtime["image_strength"],
            crf=runtime["image_crf"],
        )
    ]
    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=int(generation.get("decode_spatial_tile", 768)),
            tile_overlap_in_pixels=int(generation.get("decode_spatial_overlap", 64)),
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=int(generation.get("decode_temporal_tile", 80)),
            tile_overlap_in_frames=int(generation.get("decode_temporal_overlap", 24)),
        ),
    )
    pipeline_name = str(generation.get("pipeline", "distilled")).lower()
    common_kwargs = {
        "prompt": str(row["prompt"]),
        "seed": _seed_for_row(generation, row_index),
        "height": runtime["height"],
        "width": runtime["width"],
        "num_frames": runtime["num_frames"],
        "frame_rate": runtime["frame_rate"],
        "images": images,
        "tiling_config": tiling_config,
        "enhance_prompt": runtime["enhance_prompt"],
    }
    with torch.inference_mode():
        if pipeline_name == "distilled":
            if _as_bool(generation.get("disable_audio"), default=False):
                video, audio = _generate_distilled_video_only(
                    pipeline=pipeline,
                    **common_kwargs,
                    prompt_context=prompt_contexts.get(row_index) if prompt_contexts else None,
                    single_stage=_as_bool(generation.get("single_stage"), default=False),
                    single_stage_full_resolution=_as_bool(
                        generation.get("single_stage_full_resolution"), default=False
                    ),
                )
            else:
                video, audio = pipeline(**common_kwargs)
        else:
            params = LTX_2_3_HQ_PARAMS if "hq" in pipeline_name else LTX_2_3_PARAMS
            video, audio = pipeline(
                **common_kwargs,
                negative_prompt=str(generation.get("negative_prompt", "")),
                num_inference_steps=int(generation.get("num_inference_steps", params.num_inference_steps)),
                video_guider_params=_guider_params("video", generation, params.video_guider_params),
                audio_guider_params=_guider_params("audio", generation, params.audio_guider_params),
                max_batch_size=int(generation.get("max_batch_size", 1)),
            )

        encode_video(
            video=video,
            fps=runtime["frame_rate"],
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=get_video_chunks_number(runtime["num_frames"], tiling_config),
        )
    del video, audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"LTX-2.3 encode_video did not create output: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.checkpoint_dir is None:
        raise ValueError("LTX-2.3 batch runner requires --checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    _add_ltx_repo_to_sys_path(repo_root)

    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    _install_persistent_weight_cache(
        _as_bool(generation.get("persistent_weight_cache"), default=True)
    )
    _install_persistent_streaming_cache(
        _as_bool(generation.get("persistent_streaming_cache"), default=True),
        gpu_slots=int(generation.get("streaming_gpu_slots", 1)),
        keep_gpu=_as_bool(generation.get("persistent_streaming_gpu_cache"), default=False),
    )
    runtime = _common_runtime(generation, checkpoint_dir)
    pipeline = _load_pipeline(generation, runtime)
    prompt_contexts = _preencode_prompt_contexts(
        pipeline=pipeline,
        rows=rows,
        generation=generation,
    )

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            output_path = _generate_one(
                pipeline=pipeline,
                row=row,
                row_index=row_index,
                generation=generation,
                runtime=runtime,
                prompt_contexts=prompt_contexts,
            )
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                fps=runtime["frame_rate"],
                num_frames=runtime["num_frames"],
            )
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
