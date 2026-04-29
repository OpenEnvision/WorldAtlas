from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena persistent runner for MotionCtrl.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--openclip-pretrained-path", default=None)
    parser.add_argument("--height", default=256, type=int)
    parser.add_argument("--width", default=256, type=int)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--frames", default=16, type=int)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--ddim_steps", default=50, type=int)
    parser.add_argument("--ddim_eta", default=1.0, type=float)
    parser.add_argument("--unconditional_guidance_scale", default=7.5, type=float)
    parser.add_argument("--unconditional_guidance_scale_temporal", default=None, type=float)
    parser.add_argument("--cond_T", default=800, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError("MotionCtrl request_file must contain a list under 'requests'")
    return [dict(item) for item in requests]


def _patch_openclip(openclip_pretrained_path: str | None) -> None:
    import open_clip

    original = open_clip.create_model_and_transforms

    def patched_create_model_and_transforms(model_name, *args, **kwargs):
        if str(model_name).startswith("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
            kwargs = dict(kwargs)
            kwargs.pop("pretrained", None)
            if openclip_pretrained_path:
                return original("ViT-H-14", *args, pretrained=openclip_pretrained_path, **kwargs)
            return original("ViT-H-14", *args, pretrained=None, **kwargs)
        return original(model_name, *args, **kwargs)

    open_clip.create_model_and_transforms = patched_create_model_and_transforms


def _save_motionctrl_video(samples, output_path: Path, *, fps: float) -> None:
    import imageio
    import numpy as np
    import torch
    import torchvision

    video_fps = int(round(float(fps)))
    video = samples.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    tmp_output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(tmp_output),
        np.ascontiguousarray(grid.numpy()),
        fps=video_fps,
        macro_block_size=None,
        quality=10,
    )
    if output_path.exists():
        output_path.unlink()
    shutil.move(str(tmp_output), str(output_path))


class MotionCtrlRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        config_path: Path,
        ckpt_path: Path,
        openclip_pretrained_path: str | None,
        height: int,
        width: int,
        frames: int,
        device_id: int | None = None,
        device: str | None = None,
    ) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.repo_root = repo_root.expanduser().resolve()
        self.config_path = config_path.expanduser().resolve()
        self.ckpt_path = ckpt_path.expanduser().resolve()
        self.height = int(height)
        self.width = int(width)
        self.frames = int(frames)

        os.chdir(self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        _patch_openclip(openclip_pretrained_path)

        import torch
        from omegaconf import OmegaConf
        from main.evaluation.motionctrl_inference import load_model_checkpoint, motionctrl_sample
        from utils.utils import instantiate_from_config

        self.torch = torch
        self.motionctrl_sample = motionctrl_sample
        if device is not None:
            self.device = torch.device(device)
        elif device_id is not None:
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            capability = torch.cuda.get_device_capability(self.device)
            required_arch = f"sm_{capability[0]}{capability[1]}"
            available_arches = set(torch.cuda.get_arch_list())
            if required_arch not in available_arches:
                raise RuntimeError(
                    "MotionCtrl CUDA runtime is incompatible with this GPU: "
                    f"device={torch.cuda.get_device_name(self.device)} capability={required_arch}, "
                    f"torch={torch.__version__}, cuda={torch.version.cuda}, "
                    f"available_arches={sorted(available_arches)}"
                )

        config = OmegaConf.load(str(self.config_path))
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model = load_model_checkpoint(model, str(self.ckpt_path))
        model.eval()
        self.model = model

        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError("MotionCtrl image size [height,width] must be multiples of 16")
        channels = int(model.channels)
        model_frames = int(getattr(model, "temporal_length", self.frames))
        if self.frames != model_frames:
            raise ValueError(f"MotionCtrl config requests {self.frames} frames, model uses {model_frames}")
        self.noise_shape = [1, channels, model_frames, self.height // 8, self.width // 8]

    def generate_one(
        self,
        request: dict[str, Any],
        *,
        seed: int,
        fps: float,
        n_samples: int,
        ddim_steps: int,
        ddim_eta: float,
        unconditional_guidance_scale: float,
        unconditional_guidance_scale_temporal: float | None,
        cond_T: int,
    ) -> dict[str, Any]:
        import numpy as np

        torch = self.torch
        sample_id = str(request["sample_id"])
        output_path = Path(str(request["output_path"])).expanduser().resolve()
        camera_pose_file = Path(str(request["camera_pose_file"])).expanduser().resolve()
        prompt = str(request["prompt"]).strip()
        if not prompt:
            raise ValueError(f"MotionCtrl requires non-empty prompt for sample {sample_id}")

        pose = np.asarray(json.loads(camera_pose_file.read_text(encoding="utf-8")), dtype=np.float32)
        if pose.shape != (self.frames, 12):
            raise ValueError(
                f"MotionCtrl camera pose must have shape ({self.frames}, 12), got {tuple(pose.shape)}"
            )
        camera_poses = torch.as_tensor(pose, dtype=torch.float32, device=self.device).unsqueeze(0)

        from pytorch_lightning import seed_everything

        seed_everything(int(seed), workers=True)
        with torch.no_grad():
            batch_samples = self.motionctrl_sample(
                self.model,
                [prompt],
                self.noise_shape,
                camera_poses=camera_poses,
                trajs=None,
                n_samples=int(n_samples),
                unconditional_guidance_scale=float(unconditional_guidance_scale),
                unconditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                ddim_steps=int(ddim_steps),
                ddim_eta=float(ddim_eta),
                cond_T=int(cond_T),
            )
        samples = batch_samples[0]
        _save_motionctrl_video(samples, output_path, fps=float(fps))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        if not output_path.exists():
            raise FileNotFoundError(f"MotionCtrl output was not written: {output_path}")
        return {
            "sample_id": sample_id,
            "status": "generated",
            "prediction_path": str(output_path),
            "output_path": str(output_path),
            "prompt": prompt,
            "prompt_source": request.get("prompt_source"),
            "prompt_current": request.get("prompt_current"),
            "prompt_target": request.get("prompt_target"),
            "camera_path": list(request.get("camera_path", [])),
            "camera_pose_file": str(camera_pose_file),
            "control_source": request.get("control_source", "camera_pose_json"),
            "image_conditioning_contract": request.get(
                "image_conditioning_contract",
                "text_camera_only_no_image_conditioning",
            ),
            "fps": float(fps),
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
            "seed": int(seed),
            "ddim_steps": int(ddim_steps),
            "ddim_eta": float(ddim_eta),
            "unconditional_guidance_scale": float(unconditional_guidance_scale),
            "cond_T": int(cond_T),
        }

    def generate_many(
        self,
        requests: list[dict[str, Any]],
        *,
        seed: int,
        fps: float,
        n_samples: int,
        ddim_steps: int,
        ddim_eta: float,
        unconditional_guidance_scale: float,
        unconditional_guidance_scale_temporal: float | None,
        cond_T: int,
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        if len(requests) == 1:
            return [
                self.generate_one(
                    requests[0],
                    seed=seed,
                    fps=fps,
                    n_samples=n_samples,
                    ddim_steps=ddim_steps,
                    ddim_eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                    cond_T=cond_T,
                )
            ]

        import numpy as np
        from pytorch_lightning import seed_everything

        torch = self.torch
        prompts: list[str] = []
        output_paths: list[Path] = []
        camera_paths: list[list[str]] = []
        camera_pose_files: list[Path] = []
        pose_payloads: list[np.ndarray] = []
        for request in requests:
            sample_id = str(request["sample_id"])
            prompt = str(request["prompt"]).strip()
            if not prompt:
                raise ValueError(f"MotionCtrl requires non-empty prompt for sample {sample_id}")
            camera_pose_file = Path(str(request["camera_pose_file"])).expanduser().resolve()
            pose = np.asarray(json.loads(camera_pose_file.read_text(encoding="utf-8")), dtype=np.float32)
            if pose.shape != (self.frames, 12):
                raise ValueError(
                    f"MotionCtrl camera pose must have shape ({self.frames}, 12), got {tuple(pose.shape)}"
                )
            prompts.append(prompt)
            output_paths.append(Path(str(request["output_path"])).expanduser().resolve())
            camera_paths.append(list(request.get("camera_path", [])))
            camera_pose_files.append(camera_pose_file)
            pose_payloads.append(pose)

        camera_poses = torch.as_tensor(
            np.stack(pose_payloads, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        noise_shape = [len(requests), *self.noise_shape[1:]]
        seed_everything(int(seed), workers=True)
        with torch.no_grad():
            batch_samples = self.motionctrl_sample(
                self.model,
                prompts,
                noise_shape,
                camera_poses=camera_poses,
                trajs=None,
                n_samples=int(n_samples),
                unconditional_guidance_scale=float(unconditional_guidance_scale),
                unconditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                ddim_steps=int(ddim_steps),
                ddim_eta=float(ddim_eta),
                cond_T=int(cond_T),
            )
        results: list[dict[str, Any]] = []
        for index, (request, samples, output_path, camera_path, camera_pose_file, prompt) in enumerate(
            zip(requests, batch_samples, output_paths, camera_paths, camera_pose_files, prompts)
        ):
            _save_motionctrl_video(samples, output_path, fps=float(fps))
            if not output_path.exists():
                raise FileNotFoundError(f"MotionCtrl output was not written: {output_path}")
            results.append(
                {
                    "sample_id": str(request["sample_id"]),
                    "status": "generated",
                    "prediction_path": str(output_path),
                    "output_path": str(output_path),
                    "prompt": prompt,
                    "prompt_source": request.get("prompt_source"),
                    "prompt_current": request.get("prompt_current"),
                    "prompt_target": request.get("prompt_target"),
                    "camera_path": camera_path,
                    "camera_pose_file": str(camera_pose_file),
                    "control_source": request.get("control_source", "camera_pose_json"),
                    "image_conditioning_contract": request.get(
                        "image_conditioning_contract",
                        "text_camera_only_no_image_conditioning",
                    ),
                    "fps": float(fps),
                    "frames": self.frames,
                    "height": self.height,
                    "width": self.width,
                    "seed": int(seed) + index,
                    "ddim_steps": int(ddim_steps),
                    "ddim_eta": float(ddim_eta),
                    "unconditional_guidance_scale": float(unconditional_guidance_scale),
                    "cond_T": int(cond_T),
                }
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        return results


def run_requests(args: argparse.Namespace, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime = MotionCtrlRuntime(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config_path),
        ckpt_path=Path(args.ckpt_path),
        openclip_pretrained_path=args.openclip_pretrained_path,
        height=args.height,
        width=args.width,
        frames=args.frames,
        device=args.device,
    )

    results: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(requests), batch_size):
        batch_requests = requests[batch_start : batch_start + batch_size]
        try:
            batch_results = runtime.generate_many(
                batch_requests,
                seed=int(args.seed) + batch_start,
                fps=float(args.fps),
                n_samples=int(args.n_samples),
                ddim_steps=int(args.ddim_steps),
                ddim_eta=float(args.ddim_eta),
                unconditional_guidance_scale=float(args.unconditional_guidance_scale),
                unconditional_guidance_scale_temporal=args.unconditional_guidance_scale_temporal,
                cond_T=int(args.cond_T),
            )
            results.extend(batch_results)
        except Exception as exc:
            for request in batch_requests:
                output_path = Path(str(request.get("output_path", ""))).expanduser().resolve()
                results.append(
                    {
                        "sample_id": str(request.get("sample_id", "")),
                        "status": "failed",
                        "prediction_path": str(output_path),
                        "prompt": str(request.get("prompt", "")),
                        "camera_path": list(request.get("camera_path", [])),
                        "camera_pose_file": str(request.get("camera_pose_file", "")),
                        "control_source": request.get("control_source", "camera_pose_json"),
                        "image_conditioning_contract": request.get(
                            "image_conditioning_contract",
                            "text_camera_only_no_image_conditioning",
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
    results_file.write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0 if all(item.get("status") != "failed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
