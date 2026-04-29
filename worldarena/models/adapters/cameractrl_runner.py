from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena persistent runner for CameraCtrl.")
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--request_file", required=True)
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--ori_model_path", required=True)
    parser.add_argument("--unet_subfolder", default="unet")
    parser.add_argument("--motion_module_ckpt", required=True)
    parser.add_argument("--pose_adaptor_ckpt", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--image_lora_rank", default=2, type=int)
    parser.add_argument("--image_lora_ckpt", default=None)
    parser.add_argument("--personalized_base_model", default=None)
    parser.add_argument("--image_height", default=256, type=int)
    parser.add_argument("--image_width", default=384, type=int)
    parser.add_argument("--video_length", default=16, type=int)
    parser.add_argument("--original_pose_width", default=1280, type=int)
    parser.add_argument("--original_pose_height", default=720, type=int)
    parser.add_argument("--target_duration_sec", default=None, type=float)
    parser.add_argument("--num_inference_steps", default=25, type=int)
    parser.add_argument("--guidance_scale", default=14.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError("CameraCtrl request_file must contain a list under 'requests'")
    return [dict(item) for item in requests]


def _plucker_for_trajectory(
    *,
    trajectory_file: Path,
    image_width: int,
    image_height: int,
    original_pose_width: int,
    original_pose_height: int,
    device,
):
    import numpy as np
    import torch
    from einops import rearrange
    from inference import get_relative_pose, ray_condition
    from cameractrl.data.dataset import Camera

    with trajectory_file.open("r", encoding="utf-8") as handle:
        pose_rows = [line.strip().split(" ") for line in handle.readlines()[1:] if line.strip()]
    cam_params = [[float(value) for value in pose] for pose in pose_rows]
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    sample_wh_ratio = image_width / image_height
    pose_wh_ratio = original_pose_width / original_pose_height
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = image_height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / image_width
    else:
        resized_ori_h = image_width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / image_height
    intrinsic = np.asarray(
        [
            [
                cam_param.fx * image_width,
                cam_param.fy * image_height,
                cam_param.cx * image_width,
                cam_param.cy * image_height,
            ]
            for cam_param in cam_params
        ],
        dtype=np.float32,
    )
    K = torch.as_tensor(intrinsic)[None]
    c2ws = torch.as_tensor(get_relative_pose(cam_params))[None]
    plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device="cpu")[0]
    plucker_embedding = plucker_embedding.permute(0, 3, 1, 2).contiguous()[None].to(device)
    return rearrange(plucker_embedding, "b f c h w -> b c f h w")


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    request_file = Path(args.request_file).expanduser().resolve()
    results_file = Path(args.results_file).expanduser().resolve()
    requests = _load_requests(request_file)
    if not requests:
        results_file.write_text(json.dumps({"results": []}, indent=2), encoding="utf-8")
        return

    sys.path.insert(0, str(repo_root))
    import torch
    from omegaconf import OmegaConf
    from cameractrl.utils.util import save_videos_grid
    from inference import get_pipeline

    device = torch.device(args.device or "cuda:0")
    model_configs = OmegaConf.load(args.model_config)
    pipeline = get_pipeline(
        args.ori_model_path,
        args.unet_subfolder,
        args.image_lora_rank,
        args.image_lora_ckpt,
        model_configs.get("unet_additional_kwargs"),
        args.motion_module_ckpt,
        model_configs["pose_encoder_kwargs"],
        model_configs["attention_processor_kwargs"],
        model_configs["noise_scheduler_kwargs"],
        args.pose_adaptor_ckpt,
        args.personalized_base_model,
        str(device),
    )
    generator = torch.Generator(device=device)
    fps = 8.0
    if args.target_duration_sec and args.target_duration_sec > 0:
        fps = float(args.video_length) / float(args.target_duration_sec)

    results: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        sample_id = str(request["sample_id"])
        output_path = Path(str(request["output_path"])).expanduser().resolve()
        try:
            generator.manual_seed(int(args.seed) + index)
            plucker_embedding = _plucker_for_trajectory(
                trajectory_file=Path(str(request["trajectory_file"])).expanduser().resolve(),
                image_width=int(args.image_width),
                image_height=int(args.image_height),
                original_pose_width=int(args.original_pose_width),
                original_pose_height=int(args.original_pose_height),
                device=device,
            )
            sample = pipeline(
                prompt=str(request["prompt"]),
                negative_prompt=request.get("negative_prompt"),
                pose_embedding=plucker_embedding,
                video_length=int(args.video_length),
                height=int(args.image_height),
                width=int(args.image_width),
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                generator=generator,
            ).videos
            tmp_output = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
            tmp_output.parent.mkdir(parents=True, exist_ok=True)
            save_videos_grid(sample.cpu(), str(tmp_output), n_rows=1, fps=fps)
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(tmp_output), str(output_path))
            results.append(
                {
                    "sample_id": sample_id,
                    "status": "generated",
                    "prediction_path": str(output_path),
                    "prompt": str(request["prompt"]),
                    "camera_path": list(request.get("camera_path", [])),
                    "trajectory_file": str(request["trajectory_file"]),
                    "fps": fps,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "prediction_path": str(output_path),
                    "prompt": str(request.get("prompt", "")),
                    "error": str(exc),
                }
            )

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
