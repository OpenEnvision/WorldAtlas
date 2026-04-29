from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena batch runner for ReCamMaster.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--wan-model-dir", required=True)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=832, type=int)
    parser.add_argument("--fps", default=30.0, type=float)
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--pose_frames", default=21, type=int)
    parser.add_argument("--cfg_scale", default=5.0, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quality", default=5, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiled", action="store_true")
    return parser.parse_args()


def _load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError("ReCamMaster request_file must contain a list under 'requests'")
    return [dict(item) for item in requests]


def _wan_paths(wan_model_dir: Path) -> list[str]:
    files = [
        wan_model_dir / "diffusion_pytorch_model.safetensors",
        wan_model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        wan_model_dir / "Wan2.1_VAE.pth",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ReCamMaster missing Wan2.1 files: {missing}")
    return [str(path) for path in files]


def _resize_for_center_crop(image, *, height: int, width: int):
    import torchvision

    image_width, image_height = image.size
    scale = max(width / image_width, height / image_height)
    return torchvision.transforms.functional.resize(
        image,
        (round(image_height * scale), round(image_width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )


def _load_source_video(path: Path, *, height: int, width: int, num_frames: int):
    import imageio
    import torch
    from einops import rearrange
    from PIL import Image
    from torchvision.transforms import v2

    reader = imageio.get_reader(str(path))
    try:
        frame_total = reader.count_frames()
        if frame_total < num_frames:
            raise ValueError(f"source video needs at least {num_frames} frames, got {frame_total}: {path}")
        frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        frames = []
        for frame_id in range(num_frames):
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            frame = _resize_for_center_crop(frame, height=height, width=width)
            frames.append(frame_process(frame))
    finally:
        reader.close()
    video = torch.stack(frames, dim=0)
    video = rearrange(video, "T C H W -> 1 C T H W")
    return video


class ReCamMasterRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        ckpt_path: Path,
        wan_model_dir: Path,
        device: str | None,
    ) -> None:
        import torch
        import torch.nn as nn
        from diffsynth import ModelManager, WanVideoReCamMasterPipeline

        self.repo_root = repo_root.expanduser().resolve()
        self.ckpt_path = ckpt_path.expanduser().resolve()
        self.wan_model_dir = wan_model_dir.expanduser().resolve()
        os.chdir(self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        self.torch = torch
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(_wan_paths(self.wan_model_dir))
        pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=str(self.device))

        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

        state_dict = torch.load(str(self.ckpt_path), map_location="cpu")
        pipe.dit.load_state_dict(state_dict, strict=True)
        pipe.to(str(self.device))
        pipe.to(dtype=torch.bfloat16)
        self.pipe = pipe

    def generate_one(
        self,
        request: dict[str, Any],
        *,
        height: int,
        width: int,
        num_frames: int,
        pose_frames: int,
        fps: float,
        cfg_scale: float,
        num_inference_steps: int,
        seed: int,
        quality: int,
        tiled: bool,
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from diffsynth import save_video

        sample_id = str(request["sample_id"])
        prompt = str(request["prompt"]).strip()
        if not prompt:
            raise ValueError(f"ReCamMaster requires non-empty prompt for sample {sample_id}")
        source_video_path = Path(str(request["source_video_path"])).expanduser().resolve()
        output_path = Path(str(request["output_path"])).expanduser().resolve()
        camera_pose_file = Path(str(request["camera_pose_file"])).expanduser().resolve()
        pose = np.asarray(json.loads(camera_pose_file.read_text(encoding="utf-8")), dtype=np.float32)
        if pose.shape != (pose_frames, 12):
            raise ValueError(f"ReCamMaster camera pose must have shape ({pose_frames}, 12), got {pose.shape}")

        source_video = _load_source_video(source_video_path, height=height, width=width, num_frames=num_frames)
        target_camera = torch.as_tensor(pose, dtype=torch.bfloat16).unsqueeze(0)

        video = self.pipe(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
            source_video=source_video,
            target_camera=target_camera,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            tiled=tiled,
        )
        tmp_output = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
        tmp_output.parent.mkdir(parents=True, exist_ok=True)
        save_video(video, str(tmp_output), fps=int(round(float(fps))), quality=int(quality))
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(tmp_output), str(output_path))
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.empty_cache()
        if not output_path.exists():
            raise FileNotFoundError(f"ReCamMaster output was not written: {output_path}")
        return {
            "sample_id": sample_id,
            "status": "generated",
            "prediction_path": str(output_path),
            "output_path": str(output_path),
            "prompt": prompt,
            "prompt_source": request.get("prompt_source"),
            "prompt_current": request.get("prompt_current"),
            "prompt_target": request.get("prompt_target"),
            "source_video_path": str(source_video_path),
            "camera_path": list(request.get("camera_path", [])),
            "camera_pose_file": str(camera_pose_file),
            "control_source": request.get("control_source", "camera_pose_json"),
            "image_conditioning_contract": request.get(
                "image_conditioning_contract",
                "image_reference_frozen_to_source_video",
            ),
            "fps": float(fps),
            "num_frames": int(num_frames),
            "pose_frames": int(pose_frames),
            "height": int(height),
            "width": int(width),
            "seed": int(seed),
            "cfg_scale": float(cfg_scale),
            "num_inference_steps": int(num_inference_steps),
        }


def run_requests(args: argparse.Namespace, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime = ReCamMasterRuntime(
        repo_root=Path(args.repo_root),
        ckpt_path=Path(args.ckpt_path),
        wan_model_dir=Path(args.wan_model_dir),
        device=args.device,
    )
    results: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        try:
            results.append(
                runtime.generate_one(
                    request,
                    height=int(args.height),
                    width=int(args.width),
                    num_frames=int(args.num_frames),
                    pose_frames=int(args.pose_frames),
                    fps=float(args.fps),
                    cfg_scale=float(args.cfg_scale),
                    num_inference_steps=int(args.num_inference_steps),
                    seed=int(args.seed) + index,
                    quality=int(args.quality),
                    tiled=bool(args.tiled),
                )
            )
        except Exception as exc:
            output_path = Path(str(request.get("output_path", ""))).expanduser().resolve()
            results.append(
                {
                    "sample_id": str(request.get("sample_id", "")),
                    "status": "failed",
                    "prediction_path": str(output_path),
                    "prompt": str(request.get("prompt", "")),
                    "source_video_path": str(request.get("source_video_path", "")),
                    "camera_path": list(request.get("camera_path", [])),
                    "camera_pose_file": str(request.get("camera_pose_file", "")),
                    "control_source": request.get("control_source", "camera_pose_json"),
                    "image_conditioning_contract": request.get(
                        "image_conditioning_contract",
                        "image_reference_frozen_to_source_video",
                    ),
                    "error": str(exc) or repr(exc),
                }
            )
    return results


def main() -> int:
    args = _parse_args()
    request_file = Path(args.request_file).expanduser().resolve()
    results_file = Path(args.results_file).expanduser().resolve()
    requests = _load_requests(request_file)
    if not requests:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps({"results": []}, indent=2), encoding="utf-8")
        return 0
    results = run_requests(args, requests)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(item.get("status") != "failed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
