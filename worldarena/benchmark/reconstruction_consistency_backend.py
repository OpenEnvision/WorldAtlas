from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import numpy as np

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_project_path


DEFAULT_RECONSTRUCTION_MODEL = "depth-anything/DA3-GIANT"
DEFAULT_RECONSTRUCTION_BACKEND = "world_r1_lpips"


@dataclass(frozen=True, slots=True)
class DepthAnything3Source:
    root: Path
    package_name: str


def _default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_depth_anything3_source(
    repo_root: str | os.PathLike[str] | None = None,
) -> DepthAnything3Source:
    raw = repo_root or "thirdparty/Depth-Anything-3/src"
    resolved = resolve_project_path(raw)
    path = resolved if resolved is not None else Path(raw).expanduser()
    candidates = [
        DepthAnything3Source(path, "depth_anything_3"),
        DepthAnything3Source(path / "src", "depth_anything_3"),
        DepthAnything3Source(path, "reward_server.depth_anything_3"),
    ]
    for candidate in candidates:
        package_path = candidate.root / Path(*candidate.package_name.split("."))
        if package_path.is_dir():
            return DepthAnything3Source(candidate.root.resolve(), candidate.package_name)
    raise FileNotFoundError(f"Depth-Anything-3 source root not found: {path}")


@contextmanager
def _import_context(root: Path) -> Iterator[None]:
    root_str = str(root)
    sys.path.insert(0, root_str)
    try:
        yield
    finally:
        while root_str in sys.path:
            sys.path.remove(root_str)


@lru_cache(maxsize=4)
def _load_depth_anything3_model(
    *,
    source_root: str,
    package_name: str,
    model_name: str,
    device: str,
) -> Any:
    os.environ.update(apply_checkpoint_env())
    with _import_context(Path(source_root)):
        module = importlib.import_module(f"{package_name}.api")
        depth_anything3 = getattr(module, "DepthAnything3")

        model = depth_anything3.from_pretrained(model_name).to(device)
    model.eval()
    return model


@lru_cache(maxsize=4)
def _load_lpips_model(*, net: str, device: str) -> Any:
    try:
        import lpips
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reconstruction_consistency requires the optional `lpips` package"
        ) from exc

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    return model


def _render_size(runtime: dict[str, Any]) -> tuple[int, int]:
    configured = runtime.get("gs_render_size")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) != 2:
            raise ValueError("reconstruction_consistency gs_render_size must be [height, width]")
        return int(configured[0]), int(configured[1])
    height = int(runtime.get("render_height", 512))
    width = int(runtime.get("render_width", 512))
    return height, width


def _to_uint8_frames(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected RGB frame with shape (H, W, 3), got {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        output.append(np.ascontiguousarray(array))
    return output


def _frames_to_video_tensor(frames: Sequence[np.ndarray], *, device: str) -> Any:
    import torch

    tensors = [
        torch.from_numpy(np.ascontiguousarray(frame)).float().permute(2, 0, 1) / 255.0
        for frame in frames
    ]
    return torch.stack(tensors, dim=0).to(device)


def _generate_gs_video(
    prediction: Any,
    *,
    source_root: Path,
    package_name: str,
    render_size: tuple[int, int],
    trj_mode: str,
    chunk_size: int,
) -> Any:
    with _import_context(source_root):
        import torch

        renderer_module = importlib.import_module(f"{package_name}.model.utils.gs_renderer")
        run_renderer_in_chunk_w_trj_mode = getattr(
            renderer_module,
            "run_renderer_in_chunk_w_trj_mode",
        )

        gaussians = prediction.gaussians
        if gaussians is None:
            raise RuntimeError("Depth-Anything-3 did not return gaussian splats")

        extrinsics = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gaussians.means)
        intrinsics = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gaussians.means)
        if prediction.is_metric and prediction.scale_factor is not None:
            extrinsics[:, :, :3, 3] /= prediction.scale_factor

        color, _ = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=render_size,
            chunk_size=chunk_size,
            trj_mode=trj_mode,
            use_sh=True,
            color_mode="RGB+D",
            enable_tqdm=False,
        )
        return color.squeeze(0).clamp(0, 1)


def _score_from_lpips_distances(values: Sequence[float]) -> dict[str, Any]:
    distances = [float(value) for value in values]
    if not distances:
        raise ValueError("LPIPS returned no per-frame distances")
    mean_distance = float(np.mean(distances))
    frame_scores = [float(np.clip(1.0 - value, 0.0, 1.0)) for value in distances]
    return {
        "raw": float(np.clip(1.0 - mean_distance, 0.0, 1.0)),
        "lpips_mean": mean_distance,
        "lpips_frame_distances": distances,
        "frame_scores": frame_scores,
    }


def _compute_lpips_reconstruction_score(
    gs_video: Any,
    input_video: Any,
    *,
    device: str,
    net: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if gs_video.dim() != 4 or input_video.dim() != 4:
        raise ValueError(
            f"expected (T, C, H, W) tensors, got {gs_video.shape} and {input_video.shape}"
        )

    frame_count = min(int(gs_video.shape[0]), int(input_video.shape[0]))
    if frame_count <= 0:
        raise ValueError("reconstruction consistency requires at least one frame")

    gs_clip = gs_video[:frame_count].float()
    input_clip = input_video[:frame_count].float()
    if input_clip.shape[-2:] != gs_clip.shape[-2:]:
        input_clip = F.interpolate(
            input_clip,
            size=gs_clip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    lpips_model = _load_lpips_model(net=net, device=device)
    with torch.inference_mode():
        distances = lpips_model(gs_clip * 2.0 - 1.0, input_clip * 2.0 - 1.0)
    values = distances.detach().cpu().float().reshape(-1).tolist()
    return _score_from_lpips_distances(values)


def compute_reconstruction_consistency(
    frames: Sequence[np.ndarray],
    *,
    prompt: str | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute World-R1 S_recon for a generated video.

    This ports the released World-R1 S_recon path: reconstruct a 3D Gaussian
    representation with Depth-Anything-3, render the input camera path back to
    video, then score reconstruction agreement as clip(1 - mean LPIPS, 0, 1).
    """
    payload = dict(runtime or {})
    backend = DEFAULT_RECONSTRUCTION_BACKEND
    uint8_frames = _to_uint8_frames(frames)
    if not uint8_frames:
        return {
            "raw": None,
            "backend": backend,
            "details": {"frame_count": 0},
            "error": "no prediction frames available for reconstruction consistency",
        }

    scorer = str(payload.get("scorer") or "lpips").lower()
    if scorer != "lpips":
        raise ValueError("reconstruction_consistency currently supports scorer='lpips'")

    device = str(payload.get("device") or _default_device())
    source = _resolve_depth_anything3_source(payload.get("repo_root"))
    model_name = str(payload.get("model_name") or DEFAULT_RECONSTRUCTION_MODEL)
    process_res = int(payload.get("process_res", 504))
    render_size = _render_size(payload)
    trj_mode = str(payload.get("trj_mode") or "extend")
    chunk_size = int(payload.get("chunk_size", 4))
    lpips_net = str(payload.get("lpips_net") or "alex")
    ref_view_strategy = str(payload.get("ref_view_strategy") or "saddle_balanced")
    use_ray_pose = bool(payload.get("use_ray_pose", False))

    model = _load_depth_anything3_model(
        source_root=str(source.root),
        package_name=source.package_name,
        model_name=model_name,
        device=device,
    )
    input_tensor = _frames_to_video_tensor(uint8_frames, device=device)

    import torch

    with torch.inference_mode():
        prediction = model.inference(
            image=uint8_frames,
            extrinsics=None,
            intrinsics=None,
            process_res=process_res,
            align_to_input_ext_scale=True,
            infer_gs=True,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            export_dir=None,
        )
        gs_video = _generate_gs_video(
            prediction,
            source_root=source.root,
            package_name=source.package_name,
            render_size=render_size,
            trj_mode=trj_mode,
            chunk_size=chunk_size,
        )
        score_payload = _compute_lpips_reconstruction_score(
            gs_video,
            input_tensor,
            device=device,
            net=lpips_net,
        )

    raw = float(score_payload["raw"])
    details = {
        "source_metric": "World-R1 S_recon",
        "scorer": scorer,
        "prompt": prompt,
        "frame_count": len(uint8_frames),
        "rendered_frame_count": int(gs_video.shape[0]),
        "render_size": list(render_size),
        "process_res": process_res,
        "trj_mode": trj_mode,
        "chunk_size": chunk_size,
        "model_name": model_name,
        "source_root": str(source.root),
        "package_name": source.package_name,
        "device": device,
        "lpips_net": lpips_net,
        "lpips_mean": round(float(score_payload["lpips_mean"]), 6),
        "lpips_frame_distances": [
            round(float(value), 6)
            for value in score_payload["lpips_frame_distances"]
        ],
        "frame_scores": [
            round(float(value), 6)
            for value in score_payload["frame_scores"]
        ],
    }
    return {
        "raw": raw,
        "backend": backend,
        "details": details,
        "error": None,
    }


__all__ = [
    "DEFAULT_RECONSTRUCTION_BACKEND",
    "DEFAULT_RECONSTRUCTION_MODEL",
    "compute_reconstruction_consistency",
]
