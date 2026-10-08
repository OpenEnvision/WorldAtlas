"""Subprocess runner for Matrix-Game 1 inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from worldatlas.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldatlas.common.progress import log_progress
from worldatlas.models.adapters.windowed_rollout import (
    plan_matrix_game_1_windows,
    stitch_windowed_frames,
)

# Hunyuan / Matrix-Game I2V templates emit one ``<image>`` token. Newer
# transformers Llava requires one token per vision feature (576 for 24x24).
_LLAVA_IMAGE_FEATURES = 576


def expand_image_token_rows(
    token_rows: list[list[int]],
    mask_rows: list[list[int]] | None,
    image_token_id: int,
    num_features: int = _LLAVA_IMAGE_FEATURES,
) -> tuple[list[list[int]], list[list[int]] | None]:
    """Repeat each image placeholder so Llava can scatter the vision grid."""
    if num_features < 1:
        raise ValueError(f"num_features must be >= 1, got {num_features}")
    expanded_ids: list[list[int]] = []
    expanded_masks: list[list[int]] | None = [] if mask_rows is not None else None
    for row_index, row in enumerate(token_rows):
        mask_row = mask_rows[row_index] if mask_rows is not None else [1] * len(row)
        if len(mask_row) != len(row):
            raise ValueError(
                f"mask length {len(mask_row)} does not match tokens {len(row)} at row {row_index}"
            )
        ids: list[int] = []
        masks: list[int] = []
        for token_id, mask in zip(row, mask_row):
            if token_id == image_token_id:
                ids.extend([image_token_id] * num_features)
                masks.extend([int(mask)] * num_features)
            else:
                ids.append(int(token_id))
                masks.append(int(mask))
        expanded_ids.append(ids)
        if expanded_masks is not None:
            expanded_masks.append(masks)
    return expanded_ids, expanded_masks


def _pad_rows(rows: list[list[int]], pad_value: int = 0) -> list[list[int]]:
    width = max((len(row) for row in rows), default=0)
    return [row + [pad_value] * (width - len(row)) for row in rows]


def expand_image_token_tensors(input_ids, attention_mask, image_token_id: int, num_features: int = _LLAVA_IMAGE_FEATURES):
    """Tensor wrapper around :func:`expand_image_token_rows`."""
    import torch

    new_ids, new_masks = expand_image_token_rows(
        input_ids.tolist(),
        None if attention_mask is None else attention_mask.tolist(),
        image_token_id,
        num_features,
    )
    tensor_ids = torch.tensor(_pad_rows(new_ids), device=input_ids.device, dtype=input_ids.dtype)
    if attention_mask is None or new_masks is None:
        return tensor_ids, attention_mask
    tensor_mask = torch.tensor(
        _pad_rows(new_masks),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return tensor_ids, tensor_mask


def patch_llava_single_image_token(model, num_features: int = _LLAVA_IMAGE_FEATURES) -> bool:
    """Expand Hunyuan's single ``<image>`` token inside Llava.forward only.

    Matrix-Game's text encoder later crops hidden states using the *unexpanded*
    input_ids plus ``image_emb_len=576``. Expanding at encode() would shift those
    crop indices twice. Wrapping the Llava module keeps crop math intact.
    """
    image_token_id = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_id is None:
        return False
    original_forward = model.forward

    def forward(*args, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            token_hits = int((input_ids == image_token_id).sum().item())
            if token_hits > 0 and token_hits % num_features != 0:
                input_ids, attention_mask = expand_image_token_tensors(
                    input_ids,
                    attention_mask,
                    int(image_token_id),
                    num_features,
                )
        return original_forward(*args, input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    model.forward = forward
    return True


def bind_teacache_forward(teacache_fn):
    """Accept the RoPE tensors the official pipeline always passes.

    Upstream ``teacache_forward`` recomputes ``freqs_cos`` / ``freqs_sin`` itself
    and does not take them as arguments. ``pipeline_matrixgame`` still forwards
    them like the real DiT ``forward``, which raises TypeError.
    """

    def forward(self, *args, freqs_cos=None, freqs_sin=None, **kwargs):
        del freqs_cos, freqs_sin
        return teacache_fn(self, *args, **kwargs)

    return forward


def wrap_flash_attn_varlen_func(fn):
    """Keep ``x, _ = flash_attn_varlen_func(...)`` valid across flash-attn versions.

    Newer flash-attn returns the output tensor alone. Unpacking that tensor
    iterates the token axis and raises ``too many values to unpack (expected 2)``.
    """

    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            first = result[0]
            second = result[1] if len(result) > 1 else None
            return first, second
        return result, None

    wrapped._worldatlas_tuple_compat = True
    return wrapped


def adapt_flash_attn3_varlen(fn):
    """Map Hunyuan's FA2 positional args onto FlashAttention-3 varlen.

    FA3 inserts ``seqused_q`` / ``seqused_k`` between the cumulative-length
    tensors and the max-seqlen scalars. ``None`` means "use cu_seqlens".
    """

    def call_fa3(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, *args, **kwargs):
        del args, kwargs
        return fn(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            None,
            None,
            max_seqlen_q,
            max_seqlen_kv,
        )

    wrapped = wrap_flash_attn_varlen_func(call_fa3)
    wrapped._worldatlas_flash_backend = "flash_attn3"
    return wrapped


def resolve_flash_attn_varlen(prefer_flash_attn3: bool = True):
    """Return an FA3-compatible varlen wrapper, or None to keep FA2."""
    if not prefer_flash_attn3:
        return None
    try:
        from flash_attn_interface import flash_attn_varlen_func as flash_attn3_varlen
    except Exception:
        return None
    return adapt_flash_attn3_varlen(flash_attn3_varlen)


def patch_flash_attn_varlen_unpack(attention_module, replacement=None) -> bool:
    """Install a 2-tuple flash-attn entry point on the Hunyuan attention module."""
    if replacement is not None:
        attention_module.flash_attn_varlen_func = replacement
        return True
    original = getattr(attention_module, "flash_attn_varlen_func", None)
    if original is None:
        return False
    if getattr(original, "_worldatlas_tuple_compat", False):
        return True
    wrapped = wrap_flash_attn_varlen_func(original)
    wrapped._worldatlas_flash_backend = "flash_attn2"
    attention_module.flash_attn_varlen_func = wrapped
    return True


def as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def mg1_window_cache_dir(output_path: Path) -> Path:
    """Sidecar directory for per-window npy. Official memory-*.mp4 stays untouched."""
    output_path = Path(output_path)
    return output_path.parent / "_mg1_windows" / output_path.stem


def mg1_window_cache_file(cache_dir: Path, window_index: int) -> Path:
    return Path(cache_dir) / f"w{window_index:02d}.npy"


def condition_frame_from_chunk(frames: np.ndarray, start: int, next_start: int | None) -> np.ndarray:
    """Pick the pixel that becomes the next window's I2V condition."""
    if next_start is None:
        return frames[-1]
    rel = int(next_start) - int(start)
    if 0 <= rel < int(frames.shape[0]):
        return frames[rel]
    return frames[-1]


def load_cached_mg1_windows(cache_dir: Path, windows: list[tuple[int, int]]) -> list[np.ndarray]:
    """Load a prefix of valid cached windows. Stop at the first gap."""
    loaded: list[np.ndarray] = []
    for index, (_start, length) in enumerate(windows):
        path = mg1_window_cache_file(cache_dir, index)
        if not path.is_file():
            break
        try:
            frames = np.load(path)
        except Exception:
            break
        if not isinstance(frames, np.ndarray) or frames.ndim < 1 or int(frames.shape[0]) != int(length):
            break
        loaded.append(np.asarray(frames))
    return loaded


def save_cached_mg1_window(cache_dir: Path, window_index: int, frames: np.ndarray) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = mg1_window_cache_file(cache_dir, window_index)
    with path.open("wb") as handle:
        np.save(handle, frames)
    return path


def clear_mg1_window_cache(cache_dir: Path) -> None:
    cache_dir = Path(cache_dir)
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)


def settings_from_generation(generation: dict) -> argparse.Namespace:
    """Build the single-sample CLI namespace from a memory/generation YAML."""
    resolution = generation.get("resolution") or [1280, 720]
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("Matrix-Game-1 generation.resolution must be a 2-item list")
    inference_steps = int(generation.get("inference_steps", 50))
    return argparse.Namespace(
        video_length=int(generation.get("video_length", 65)),
        guidance_scale=float(generation.get("guidance_scale", 6.0)),
        inference_steps=inference_steps,
        shift=float(generation.get("shift", 15.0)),
        num_pre_frames=int(generation.get("num_pre_frames", 5)),
        num_steps=int(generation.get("num_steps", inference_steps)),
        rel_l1_thresh=float(generation.get("rel_l1_thresh", 0.075)),
        resolution=[int(resolution[0]), int(resolution[1])],
        fps=int(generation.get("fps", 16)),
        seed=int(generation.get("seed", 42)),
        chunk_video_length=int(generation.get("chunk_video_length", 65)),
        bfloat16=as_bool(generation.get("bfloat16"), False),
        prefer_flash_attn3=as_bool(generation.get("prefer_flash_attn3"), True),
    )


def reset_teacache(pipeline) -> None:
    transformer_cls = pipeline.transformer.__class__
    transformer_cls.cnt = 0
    transformer_cls.accumulated_rel_l1_distance = 0
    transformer_cls.previous_modulated_input = None
    transformer_cls.previous_residual = None


def patch_matrix_game_1_text_encoder(text_enc) -> bool:
    """Patch the Llava backbone inside ``MatrixGameEncoderWrapperI2V``."""
    encoder = getattr(text_enc, "text_encoder_1", text_enc)
    model = getattr(encoder, "model", None)
    if model is None:
        return False
    return patch_llava_single_image_token(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Matrix-Game-1 runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--actions_json", required=True, type=str)
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--video_length", default=65, type=int)
    parser.add_argument("--guidance_scale", default=6.0, type=float)
    parser.add_argument("--inference_steps", default=50, type=int)
    parser.add_argument("--shift", default=15.0, type=float)
    parser.add_argument("--num_pre_frames", default=5, type=int)
    parser.add_argument(
        "--num_steps",
        default=None,
        type=int,
        help="TeaCache step budget. Defaults to --inference_steps so the two stay aligned.",
    )
    parser.add_argument("--rel_l1_thresh", default=0.075, type=float)
    parser.add_argument("--resolution", nargs=2, default=[1280, 720], type=int)
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--chunk_video_length",
        default=65,
        type=int,
        help="Official short-clip length. Longer --video_length is tiled with overlap.",
    )
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument(
        "--prefer_flash_attn3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Hopper FlashAttention-3 varlen kernel when it is installed.",
    )
    parsed = parser.parse_args()
    if parsed.num_steps is None:
        parsed.num_steps = parsed.inference_steps
    return parsed


def ensure_single_process_group() -> None:
    """ActionModule calls dist.get_world_size() during DiT init.

    Hope reload_sharded workers are one Python process per GPU and never run
    torchrun, so the default group is missing. A 1-rank gloo group is enough:
    the hybrid sequence-parallel attention stays disabled.
    """
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return
    # HashStore is in-process, so eight workers on one node cannot collide on a port.
    try:
        store = dist.HashStore()
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
        return
    except Exception:
        pass
    port = 29500 + (os.getpid() % 2000)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )


def _resize_and_crop_image(image, target_size: tuple[int, int]):
    width, height = image.size
    target_width, target_height = target_size

    if height / width > target_height / target_width:
        new_width = int(width)
        new_height = int(new_width * target_height / target_width)
    else:
        new_height = int(height)
        new_width = int(new_height * target_width / target_height)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))


def prepare_matrix_game_1_repo(repo_root: Path) -> Path:
    """chdir into the official checkout so Hunyuan relative imports resolve."""
    os.environ.update(apply_checkpoint_env())
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo_root = repo_root.expanduser().resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_matrix_game_1(checkpoint_dir: Path, args: argparse.Namespace):
    """Load DiT / VAE / text encoder once. Callers reuse this across samples."""
    import torch
    from matrixgame.encoder_variants import get_text_enc
    from matrixgame.model_variants.matrixgame_dit_src import MGVideoDiffusionTransformerI2V
    from matrixgame.model_variants.matrixgame_dit_src import attenion as attention_mod
    from matrixgame.sample.flow_matching_scheduler_matrixgame import FlowMatchDiscreteScheduler
    from matrixgame.sample.pipeline_matrixgame import MatrixGameVideoPipeline
    from matrixgame.vae_variants import get_vae
    from teacache_forward import teacache_forward

    if not torch.cuda.is_available():
        raise RuntimeError("Matrix-Game-1 requires CUDA for inference.")
    ensure_single_process_group()
    flash_replacement = resolve_flash_attn_varlen(getattr(args, "prefer_flash_attn3", True))
    patch_flash_attn_varlen_unpack(attention_mod, replacement=flash_replacement)
    flash_backend = getattr(
        attention_mod.flash_attn_varlen_func,
        "_worldatlas_flash_backend",
        "flash_attn2",
    )
    log_progress("mg1_attention", backend=flash_backend)
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16 if args.bfloat16 else torch.float32
    scheduler = FlowMatchDiscreteScheduler(
        shift=args.shift,
        reverse=True,
        solver="euler",
    )
    vae = get_vae("matrixgame", str(checkpoint_dir / "vae"), torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae.enable_tiling()

    dit = MGVideoDiffusionTransformerI2V.from_pretrained(str(checkpoint_dir / "dit"))
    dit.requires_grad_(False)
    dit.eval()

    text_enc = get_text_enc(
        "matrixgame",
        str(checkpoint_dir),
        weight_dtype=weight_dtype,
        i2v_type="refiner",
    )
    patch_matrix_game_1_text_encoder(text_enc)

    pipeline = MatrixGameVideoPipeline(
        vae=vae.vae,
        text_encoder=text_enc,
        transformer=dit,
        scheduler=scheduler,
    ).to(weight_dtype).to(device)

    transformer_cls = pipeline.transformer.__class__
    transformer_cls.enable_teacache = True
    transformer_cls.num_steps = args.num_steps
    transformer_cls.rel_l1_thresh = args.rel_l1_thresh
    transformer_cls.forward = bind_teacache_forward(teacache_forward)
    reset_teacache(pipeline)
    return {
        "pipeline": pipeline,
        "vae": vae,
        "args": args,
        "device": device,
        "weight_dtype": weight_dtype,
    }


def generate_matrix_game_1(
    runtime: dict,
    *,
    action_spec: dict,
    image_path: Path,
    output_path: Path,
    prompt: str,
) -> None:
    """Run one windowed 60s (or short-clip) sample on a resident pipeline."""
    import imageio
    import torch
    from diffusers.video_processor import VideoProcessor
    from einops import rearrange
    from PIL import Image

    args = runtime["args"]
    pipeline = runtime["pipeline"]
    vae = runtime["vae"]
    device = runtime["device"]
    condition_dtype = torch.bfloat16 if args.bfloat16 else torch.float16

    keyboard_condition = np.asarray(action_spec["keyboard_condition"], dtype=np.float32)
    mouse_condition = np.asarray(action_spec["mouse_condition"], dtype=np.float32)
    if keyboard_condition.shape[0] != args.video_length or mouse_condition.shape[0] != args.video_length:
        raise ValueError(
            "Matrix-Game-1 action sequence length mismatch: "
            f"video_length={args.video_length}, keyboard={keyboard_condition.shape[0]}, "
            f"mouse={mouse_condition.shape[0]}"
        )

    image = Image.open(image_path).convert("RGB")
    new_width, new_height = int(args.resolution[0]), int(args.resolution[1])
    condition_image = _resize_and_crop_image(image, (new_width, new_height))
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
    effective_prompt = (prompt or "").strip() or "N.o.n.e"
    windows = plan_matrix_game_1_windows(
        args.video_length,
        chunk_frames=int(args.chunk_video_length),
        overlap_frames=max(int(args.num_pre_frames), 0),
    )
    cache_dir = mg1_window_cache_dir(output_path)
    chunk_videos = load_cached_mg1_windows(cache_dir, windows)
    resume_index = len(chunk_videos)
    if resume_index:
        prev_start, _prev_length = windows[resume_index - 1]
        next_start = windows[resume_index][0] if resume_index < len(windows) else None
        condition_image = Image.fromarray(
            condition_frame_from_chunk(chunk_videos[-1], prev_start, next_start)
        ).convert("RGB")
        log_progress(
            "mg1_window_resume",
            sample=output_path.stem,
            cached=resume_index,
            windows=len(windows),
        )

    def _initial_tensor(pil_image):
        tensor = video_processor.preprocess(pil_image, height=new_height, width=new_width)
        if args.num_pre_frames > 0:
            tensor = torch.cat([tensor, tensor.repeat(args.num_pre_frames, 1, 1, 1)], dim=0)
        return tensor

    inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    for window_index, (start, length) in enumerate(windows):
        if window_index < resume_index:
            continue
        reset_teacache(pipeline)
        pipeline.transformer.__class__.num_steps = args.num_steps
        pipeline.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh
        keyboard_tensor = torch.tensor(
            keyboard_condition[start : start + length],
            dtype=condition_dtype,
            device=device,
        ).unsqueeze(0)
        mouse_tensor = torch.tensor(
            mouse_condition[start : start + length],
            dtype=condition_dtype,
            device=device,
        ).unsqueeze(0)
        if keyboard_tensor.shape[1] != length or mouse_tensor.shape[1] != length:
            raise ValueError(
                "Matrix-Game-1 chunk action slice is short: "
                f"window=({start}, {length}) keyboard={keyboard_tensor.shape} mouse={mouse_tensor.shape}"
            )
        window_args = argparse.Namespace(**vars(args))
        window_args.video_length = int(length)
        log_progress(
            "mg1_window_start",
            sample=output_path.stem,
            window=f"{window_index + 1}/{len(windows)}",
            start=start,
            length=length,
            steps=args.inference_steps,
        )
        with inference_ctx():
            video = pipeline(
                prompt=effective_prompt,
                height=new_height,
                width=new_width,
                video_length=length,
                mouse_condition=mouse_tensor,
                keyboard_condition=keyboard_tensor,
                initial_image=_initial_tensor(condition_image),
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                embedded_guidance_scale=None,
                data_type="video",
                vae_ver="884-16c-hy",
                enable_tiling=True,
                generator=torch.Generator(device="cuda").manual_seed(args.seed + window_index),
                i2v_type="refiner",
                args=window_args,
                semantic_images=condition_image,
            ).videos[0]
        frames = rearrange(video.permute(1, 0, 2, 3) * 255, "t c h w -> t h w c").contiguous()
        frames = frames.cpu().numpy().astype(np.uint8)
        save_cached_mg1_window(cache_dir, window_index, frames)
        chunk_videos.append(frames)
        next_start = windows[window_index + 1][0] if window_index + 1 < len(windows) else None
        if next_start is not None:
            condition_image = Image.fromarray(
                condition_frame_from_chunk(frames, start, next_start)
            ).convert("RGB")
        log_progress(
            "mg1_window_done",
            sample=output_path.stem,
            window=f"{window_index + 1}/{len(windows)}",
            frames=int(frames.shape[0]),
        )

    frames = stitch_windowed_frames(
        chunk_videos,
        windows,
        target_frames=args.video_length,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=args.fps)
    clear_mg1_window_cache(cache_dir)
    reset_teacache(pipeline)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd()
    repo_root = prepare_matrix_game_1_repo((launch_cwd / Path(args.repo_root).expanduser()).resolve())
    del repo_root
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Matrix-Game-1 runner requires checkpoint_dir")
    actions_path = (launch_cwd / Path(args.actions_json).expanduser()).resolve()
    image_path = (launch_cwd / Path(args.image_path).expanduser()).resolve()
    output_path = (launch_cwd / Path(args.output_path).expanduser()).resolve()
    with actions_path.open("r", encoding="utf-8") as file:
        action_spec = json.load(file)
    runtime = load_matrix_game_1(checkpoint_dir, args)
    generate_matrix_game_1(
        runtime,
        action_spec=action_spec,
        image_path=image_path,
        output_path=output_path,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
