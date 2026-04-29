from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infinite-World single-sample runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--actions_json", default=None, type=str)
    parser.add_argument("--image_path", default=None, type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--requests_json", default=None, type=str)
    parser.add_argument("--results_json", default=None, type=str)
    parser.add_argument("--config_path", default="configs/infworld_config.yaml", type=str)
    parser.add_argument("--bucket_config_name", default="ASPECT_RATIO_627_F64", type=str)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    parser.add_argument("--num_sampling_steps", default=30, type=int)
    parser.add_argument("--sample_shift", default=None, type=float)
    parser.add_argument("--sample_guide_scale", default=None, type=float)
    parser.add_argument("--quality", default=8, type=int)
    parser.add_argument("--target_duration_sec", default=None, type=float)
    parser.add_argument("--negative_prompt", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    return parser.parse_args()


def _load_torch_state_dict(path: str, *, map_location: str):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_path(path_value: str | None, *, base_dir: Path) -> str | None:
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _setup_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _set_single_process_context_parallel(cp_util_module) -> None:
    cp_util_module.dp_rank = 0
    cp_util_module.dp_size = 1
    cp_util_module.cp_rank = 0
    cp_util_module.cp_size = 1
    cp_util_module.dp_group = None
    cp_util_module.cp_group = None


def _resize_and_center_crop(frame: np.ndarray, target_size: tuple[int, int]):
    import torch
    from torchvision.transforms import functional as tf

    target_h, target_w = target_size
    orig_h, orig_w = frame.shape[:2]
    scale = max(target_h / orig_h, target_w / orig_w)
    resized_h = int(math.ceil(scale * orig_h))
    resized_w = int(math.ceil(scale * orig_w))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    cropped = tf.center_crop(tensor, [target_h, target_w])
    return cropped.permute(1, 2, 0).contiguous().numpy()


def _load_condition_video(image_path: Path, bucket_config) -> Any:
    import torch

    frame = _read_condition_frame(image_path)
    target_h, target_w = _bucket_hw_for_frame(frame, bucket_config)
    cropped = _resize_and_center_crop(frame, (int(target_h), int(target_w)))
    tensor = torch.from_numpy(cropped).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0
    return tensor.unsqueeze(0).unsqueeze(2)


def _read_condition_frame(image_path: Path) -> np.ndarray:
    if image_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(image_path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"Failed to decode condition video: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Failed to read condition image: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def _bucket_hw_for_frame(frame: np.ndarray, bucket_config) -> tuple[int, int]:
    aspect_ratio = frame.shape[0] / frame.shape[1]
    closest_bucket = min(bucket_config, key=lambda key: abs(float(key) - aspect_ratio))
    target_h, target_w = bucket_config[closest_bucket][0]
    return int(target_h), int(target_w)


def _request_bucket_key(request: dict[str, Any], *, bucket_config) -> tuple[int, int]:
    return _bucket_hw_for_frame(_read_condition_frame(Path(str(request["image_path"]))), bucket_config)


def _load_action_sequence(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_actions = payload.get("frame_actions", payload) if isinstance(payload, dict) else payload
    move_action_map = {
        "no-op": 0,
        "go forward": 1,
        "go back": 2,
        "go left": 3,
        "go right": 4,
        "go forward and go left": 5,
        "go forward and go right": 6,
        "go back and go left": 7,
        "go back and go right": 8,
        "uncertain": 9,
    }
    view_action_map = {
        "no-op": 0,
        "turn up": 1,
        "turn down": 2,
        "turn left": 3,
        "turn right": 4,
        "turn up and turn left": 5,
        "turn up and turn right": 6,
        "turn down and turn left": 7,
        "turn down and turn right": 8,
        "uncertain": 9,
    }
    move_ids = [move_action_map[str(item["move"])] for item in frame_actions]
    view_ids = [view_action_map[str(item["view"])] for item in frame_actions]
    return move_ids, view_ids


def _action_slice(ids, *, start: int, frames: int):
    import torch

    sliced = ids[start : start + frames]
    if sliced.shape[0] < frames:
        pad_len = frames - sliced.shape[0]
        sliced = torch.cat([sliced, torch.zeros(pad_len, dtype=torch.long)], dim=0)
    return sliced


def _write_video(output_path: Path, video_tensor, *, fps: int, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = video_tensor.detach().cpu().float()[0]
    frames = tensor.permute(1, 2, 3, 0)
    frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0)
    frames = (frames.numpy() * 255.0).round().clip(0.0, 255.0).astype(np.uint8)
    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=quality,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _request_payloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.requests_json:
        payload = json.loads(Path(args.requests_json).read_text(encoding="utf-8"))
        requests = payload.get("requests", payload) if isinstance(payload, dict) else payload
        if not isinstance(requests, list):
            raise ValueError("Infinite-World requests_json must contain a list or a {'requests': list} payload")
        return [dict(item) for item in requests]

    missing = [
        name
        for name in ("actions_json", "image_path", "prompt", "output_path")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Single-sample Infinite-World mode requires "
            + ", ".join(f"--{name}" for name in missing)
        )
    return [
        {
            "sample_id": "single",
            "actions_json": args.actions_json,
            "image_path": args.image_path,
            "prompt": args.prompt,
            "output_path": args.output_path,
            "seed": args.seed,
        }
    ]


def _generate_request(
    request: dict[str, Any],
    *,
    args: argparse.Namespace,
    device,
    vae,
    text_encoder,
    scheduler,
    dit,
    bucket_config,
    validation_num_frames: int,
    guidance_scale: float,
    negative_prompt: str,
) -> None:
    import torch

    _setup_seed(int(request.get("seed", args.seed)))
    condition_video = _load_condition_video(Path(str(request["image_path"])), bucket_config).to(device)
    move_ids, view_ids = _load_action_sequence(Path(str(request["actions_json"])))
    move_ids = torch.tensor(move_ids, dtype=torch.long)
    view_ids = torch.tensor(view_ids, dtype=torch.long)
    video_buffer = condition_video.detach().cpu().float()

    for _ in range(max(int(args.num_chunks), 1)):
        current_cond = video_buffer.to(device, dtype=torch.float32)
        current_latent = vae.encode(current_cond)

        latent_size = list(current_latent.shape)
        latent_size[2] = 21
        curr_start = int(video_buffer.shape[2] - 1)
        curr_end = curr_start + validation_num_frames
        move = move_ids[curr_start:curr_end]
        view = view_ids[curr_start:curr_end]
        if move.shape[0] < validation_num_frames:
            pad_len = validation_num_frames - move.shape[0]
            move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        additional_args = {
            "image_cond": current_latent,
            "move": move.unsqueeze(0).to(device),
            "view": view.unsqueeze(0).to(device),
        }
        with torch.no_grad():
            samples = scheduler.sample(
                model=dit,
                text_encoder=text_encoder,
                null_embedder=dit.y_embedder,
                z_size=torch.Size(latent_size),
                prompts=[str(request["prompt"])],
                guidance_scale=guidance_scale,
                negative_prompts=[negative_prompt],
                device=device,
                additional_args=additional_args,
            )
            decoded_chunk = vae.decode(samples).detach().cpu()
            video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
        torch.cuda.empty_cache()

    if args.target_duration_sec is not None:
        target_frames = max(int(round(float(args.target_duration_sec) * float(args.fps))), 1)
        video_buffer = video_buffer[:, :, :target_frames]

    _write_video(Path(str(request["output_path"])).expanduser().resolve(), video_buffer, fps=args.fps, quality=args.quality)


def _prepare_request(request: dict[str, Any], *, bucket_config) -> dict[str, Any]:
    import torch

    move_ids, view_ids = _load_action_sequence(Path(str(request["actions_json"])))
    return {
        "request": request,
        "condition_video": _load_condition_video(Path(str(request["image_path"])), bucket_config),
        "move_ids": torch.tensor(move_ids, dtype=torch.long),
        "view_ids": torch.tensor(view_ids, dtype=torch.long),
    }


def _write_prepared_batch_videos(prepared: list[dict[str, Any]], video_buffer, *, args: argparse.Namespace) -> None:
    if args.target_duration_sec is not None:
        target_frames = max(int(round(float(args.target_duration_sec) * float(args.fps))), 1)
        video_buffer = video_buffer[:, :, :target_frames]
    for index, item in enumerate(prepared):
        request = item["request"]
        _write_video(
            Path(str(request["output_path"])).expanduser().resolve(),
            video_buffer[index : index + 1],
            fps=args.fps,
            quality=args.quality,
        )


def _generate_prepared_batch(
    prepared: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device,
    vae,
    text_encoder,
    scheduler,
    dit,
    validation_num_frames: int,
    guidance_scale: float,
    negative_prompt: str,
) -> None:
    import torch

    if not prepared:
        return
    _setup_seed(int(prepared[0]["request"].get("seed", args.seed)))
    video_buffer = torch.cat([item["condition_video"] for item in prepared], dim=0).detach().cpu().float()
    prompts = [str(item["request"]["prompt"]) for item in prepared]

    for _ in range(max(int(args.num_chunks), 1)):
        current_cond = video_buffer.to(device, dtype=torch.float32)
        current_latent = vae.encode(current_cond)

        latent_size = list(current_latent.shape)
        latent_size[2] = 21
        curr_start = int(video_buffer.shape[2] - 1)
        move = torch.stack(
            [
                _action_slice(item["move_ids"], start=curr_start, frames=validation_num_frames)
                for item in prepared
            ],
            dim=0,
        )
        view = torch.stack(
            [
                _action_slice(item["view_ids"], start=curr_start, frames=validation_num_frames)
                for item in prepared
            ],
            dim=0,
        )

        additional_args = {
            "image_cond": current_latent,
            "move": move.to(device),
            "view": view.to(device),
        }
        with torch.no_grad():
            samples = scheduler.sample(
                model=dit,
                text_encoder=text_encoder,
                null_embedder=dit.y_embedder,
                z_size=torch.Size(latent_size),
                prompts=prompts,
                guidance_scale=guidance_scale,
                negative_prompts=[negative_prompt],
                device=device,
                additional_args=additional_args,
            )
            decoded_chunk = vae.decode(samples).detach().cpu()
            video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
        torch.cuda.empty_cache()

    _write_prepared_batch_videos(prepared, video_buffer, args=args)


def _run_generation_requests(
    requests: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device,
    vae,
    text_encoder,
    scheduler,
    dit,
    bucket_config,
    validation_num_frames: int,
    guidance_scale: float,
    negative_prompt: str,
) -> list[dict[str, Any]]:
    import torch

    batch_size = max(int(args.batch_size), 1)
    results: list[dict[str, Any]] = []
    grouped_requests: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for request in requests:
        key = _request_bucket_key(request, bucket_config=bucket_config)
        grouped_requests.setdefault(key, []).append(request)

    def append_success(items: list[dict[str, Any]]) -> None:
        for item in items:
            request = item["request"]
            results.append(
                {
                    "status": "generated",
                    "sample_id": str(request.get("sample_id", len(results) + 1)),
                    "prediction_path": str(Path(str(request["output_path"])).expanduser().resolve()),
                    "prompt": str(request["prompt"]),
                    "prompt_source": request.get("prompt_source", "prompt_target"),
                    "prompt_current": request.get("prompt_current"),
                    "prompt_target": request.get("prompt_target"),
                    "camera_path": request.get("camera_path", []),
                    "camera_tokens": request.get("camera_tokens", []),
                    "actions": request.get("actions", []),
                    "action_spec_path": request.get("action_spec_path", request.get("actions_json")),
                    "action_source": request.get("action_source"),
                    "control_source": request.get("control_source"),
                    "batch_size": len(items),
                }
            )

    for grouped in grouped_requests.values():
        for start in range(0, len(grouped), batch_size):
            group = [
                _prepare_request(request, bucket_config=bucket_config)
                for request in grouped[start : start + batch_size]
            ]

            try:
                _generate_prepared_batch(
                    group,
                    args=args,
                    device=device,
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    dit=dit,
                    validation_num_frames=validation_num_frames,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )
                append_success(group)
            except RuntimeError as exc:
                if len(group) == 1 or "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                for item in group:
                    _generate_prepared_batch(
                        [item],
                        args=args,
                        device=device,
                        vae=vae,
                        text_encoder=text_encoder,
                        scheduler=scheduler,
                        dit=dit,
                        validation_num_frames=validation_num_frames,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                    )
                    append_success([item])

    return results


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.ckpt_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Infinite-World runner requires ckpt_dir")
    os.environ.update(apply_checkpoint_env())
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Infinite-World requires a CUDA-enabled runtime.")

    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
    _setup_seed(args.seed)

    import infworld.context_parallel.context_parallel_util as cp_util
    from infworld.configs import bucket_config as bucket_config_module
    from infworld.utils.prepare_dataloader import get_obj_from_str

    _set_single_process_context_parallel(cp_util)

    config_path = Path(_resolve_path(args.config_path, base_dir=repo_root) or "")
    cfg = OmegaConf.load(config_path)
    cfg.checkpoint_path = _resolve_path(cfg.get("checkpoint_path"), base_dir=checkpoint_dir)
    if "vae_cfg" in cfg and "vae_pth" in cfg.vae_cfg:
        cfg.vae_cfg.vae_pth = _resolve_path(cfg.vae_cfg.vae_pth, base_dir=checkpoint_dir)
    if "text_encoder_cfg" in cfg:
        if "checkpoint_path" in cfg.text_encoder_cfg:
            cfg.text_encoder_cfg.checkpoint_path = _resolve_path(
                cfg.text_encoder_cfg.checkpoint_path,
                base_dir=checkpoint_dir,
            )
        if "tokenizer_path" in cfg.text_encoder_cfg:
            cfg.text_encoder_cfg.tokenizer_path = _resolve_path(
                cfg.text_encoder_cfg.tokenizer_path,
                base_dir=checkpoint_dir,
            )

    vae = get_obj_from_str(cfg.vae_target)(**cfg.vae_cfg).to(device)
    text_encoder = get_obj_from_str(cfg.text_encoder_target)(device=device.index or 0, **cfg.text_encoder_cfg)
    text_encoder.t5.model.to(device)

    scheduler = get_obj_from_str(cfg.scheduler_target)(**cfg.val_scheduler_cfg)
    scheduler.num_sampling_steps = int(args.num_sampling_steps)
    scheduler.shift = float(
        args.sample_shift if args.sample_shift is not None else cfg.val_scheduler_cfg.get("shift", 7.0)
    )

    model_dtype = getattr(torch, cfg.get("amp_dtype", "bfloat16"))
    dit = get_obj_from_str(cfg.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=False,
        **cfg.model_cfg,
    )
    dit = dit.to(dtype=model_dtype)
    dit.eval()

    state_dict = _load_torch_state_dict(str(cfg.checkpoint_path), map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    dit.load_state_dict(state_dict, strict=False)
    dit = dit.to(device)

    bucket_config = getattr(bucket_config_module, args.bucket_config_name)
    validation_num_frames = int(cfg.validation_data.get("num_frames", 81))
    guidance_scale = float(
        args.sample_guide_scale
        if args.sample_guide_scale is not None
        else cfg.val_scheduler_cfg.get("text_cfg_scale", 5.0)
    )
    negative_prompt = (
        args.negative_prompt
        or "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, "
        "image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
        "motionless image, cluttered background, three legs, crowded background, walking backwards."
    )

    requests = _request_payloads(args)
    try:
        results = _run_generation_requests(
            requests,
            args=args,
            device=device,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            dit=dit,
            bucket_config=bucket_config,
            validation_num_frames=validation_num_frames,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )
    except Exception as exc:
        if not args.requests_json:
            raise
        results = [
            {
                "status": "failed",
                "sample_id": str(request.get("sample_id", index)),
                "prediction_path": str(request.get("output_path", "")),
                "prompt": str(request.get("prompt", "")),
                "error": f"{type(exc).__name__}: {exc}",
            }
            for index, request in enumerate(requests, start=1)
        ]

    if args.results_json:
        results_path = Path(args.results_json).expanduser().resolve()
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
