"""Reconstruction-consistency backend aligned with WBench."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib
import locale
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

from worldarena.common.checkpoints import apply_checkpoint_env, hf_local_dir, resolve_project_path


DEFAULT_RECONSTRUCTION_MODEL = "depth-anything/DA3-GIANT-1.1"
DEFAULT_RECONSTRUCTION_BACKEND = "da3_reprojection"
DEFAULT_MAX_PTS_PER_FRAME = 5000
DEFAULT_RECONSTRUCTION_FPS = 3.0


@dataclass(frozen=True, slots=True)
class DepthAnything3Source:
    """Resolved source root for Depth-Anything-3."""

    root: Path
    package_name: str


def _default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _prepend_process_path(key: str, path: Path) -> None:
    if not path.exists():
        return
    current = os.environ.get(key)
    values = [str(path)]
    if current:
        values.append(current)
    os.environ[key] = os.pathsep.join(dict.fromkeys(values))


def _ensure_cuda_toolkit_env() -> None:
    os.environ["LANG"] = "C.UTF-8"
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        locale.setlocale(locale.LC_ALL, os.environ["LC_ALL"])
    except locale.Error:
        pass
    _prepend_process_path("PATH", Path(sys.executable).resolve().parent)
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and (Path(cuda_home) / "bin" / "nvcc").exists():
        selected = Path(cuda_home)
    else:
        candidates: list[Path] = []
        try:
            import torch

            if torch.version.cuda:
                major, minor = torch.version.cuda.split(".")[:2]
                candidates.append(Path(f"/usr/local/cuda-{major}.{minor}"))
        except Exception:
            pass
        candidates.extend(
            [
                Path("/usr/local/cuda-12.4"),
                Path("/usr/local/cuda-12.8"),
                Path("/usr/local/cuda-12.9"),
                Path("/usr/local/cuda-12.1"),
                Path("/usr/local/cuda-12"),
                Path("/usr/local/cuda"),
            ]
        )
        selected = next(
            (candidate for candidate in candidates if (candidate / "bin" / "nvcc").exists()),
            None,
        )
        if selected is None:
            return
        os.environ["CUDA_HOME"] = str(selected)
        os.environ.setdefault("CUDA_PATH", str(selected))

    _prepend_process_path("PATH", selected / "bin")
    _prepend_process_path("LD_LIBRARY_PATH", selected / "lib64")
    _patch_torch_cpp_extension_utf8_hash()


def _patch_torch_cpp_extension_utf8_hash() -> None:
    try:
        import torch.utils._cpp_extension_versioner as versioner
    except Exception:
        return
    if getattr(versioner.hash_source_files, "_worldarena_utf8_patch", False):
        return

    def hash_source_files_utf8(hash_value: int, source_files: list[str]) -> int:
        for filename in source_files:
            with open(filename, encoding="utf-8") as file:
                hash_value = versioner.update_hash(hash_value, file.read())
        return hash_value

    hash_source_files_utf8._worldarena_utf8_patch = True  # type: ignore[attr-defined]
    versioner.hash_source_files = hash_source_files_utf8


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


def _resolve_model_name(model_name: str) -> str:
    token = str(model_name).strip()
    if not token:
        return token
    if token.startswith(("ckpt/", "./", "../", "/", "~")):
        resolved = resolve_project_path(token)
        return str(resolved if resolved is not None else Path(token).expanduser())
    if "/" in token:
        local = hf_local_dir(token, required=False)
        if local.is_dir():
            return str(local)
    return token


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


def extract_wbench_frames(
    video_path: str | os.PathLike[str],
    *,
    fps: float = DEFAULT_RECONSTRUCTION_FPS,
    decode_threads: int | None = None,
    decode_mode: str = "random_seek",
) -> tuple[list[np.ndarray], list[int], dict[str, Any]]:
    """Decode frames with the exact sampling rule used by WBench's DA3 precompute."""
    if decode_threads is not None and decode_threads > 0 and hasattr(cv2, "CAP_PROP_N_THREADS"):
        capture = cv2.VideoCapture(
            str(video_path),
            cv2.CAP_ANY,
            [cv2.CAP_PROP_N_THREADS, int(decode_threads)],
        )
    else:
        capture = cv2.VideoCapture(str(video_path))
    actual_decode_threads = (
        int(capture.get(cv2.CAP_PROP_N_THREADS))
        if hasattr(cv2, "CAP_PROP_N_THREADS")
        else None
    )
    try:
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps <= 0 or total_frames <= 0:
            raise ValueError(
                f"failed to probe video for WBench sampling: {video_path} "
                f"(fps={video_fps}, frames={total_frames})"
            )

        if fps <= 0 or fps >= video_fps:
            step = 1
        else:
            step = max(1, int(round(video_fps / fps)))
        requested_indices = list(range(0, total_frames, step))
        frames: list[np.ndarray] = []
        if decode_mode == "sequential_exact":
            next_position = 0
            for index in range(requested_indices[-1] + 1):
                success, frame_bgr = capture.read()
                if not success:
                    break
                if index == requested_indices[next_position]:
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    next_position += 1
                    if next_position == len(requested_indices):
                        break
        elif decode_mode == "random_seek":
            for index in requested_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame_bgr = capture.read()
                if success:
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(
                "WBench frame decode_mode must be 'random_seek' or 'sequential_exact'"
            )
    finally:
        capture.release()

    # WBench returns this prefix even if a seek/read in the middle failed.
    frame_indices = requested_indices[: len(frames)]
    return frames, frame_indices, {
        "frame_source": "wbench_fps",
        "frame_sampling_fps": float(fps),
        "source_video_fps": video_fps,
        "source_frame_count": total_frames,
        "frame_step": step,
        "frame_indices": frame_indices,
        "decode_threads": actual_decode_threads,
        "decode_mode": decode_mode,
    }


def _resize_rgb(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def _as_w2c_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    array = np.asarray(extrinsics, dtype=np.float64)
    if array.ndim == 3 and array.shape[1:] == (3, 4):
        ext_44 = np.zeros((array.shape[0], 4, 4), dtype=np.float64)
        ext_44[:, :3, :4] = array
        ext_44[:, 3, 3] = 1.0
        return ext_44
    if array.ndim == 3 and array.shape[1:] == (4, 4):
        return array
    raise ValueError(f"unsupported extrinsics shape: {array.shape}")


def _as_intrinsics_33(intrinsics: np.ndarray) -> np.ndarray:
    array = np.asarray(intrinsics, dtype=np.float64)
    if array.ndim == 3 and array.shape[1:] == (3, 3):
        return array
    if array.ndim == 3 and array.shape[1:] == (4, 4):
        return array[:, :3, :3]
    if array.ndim == 2 and array.shape == (3, 3):
        return array[None, ...]
    raise ValueError(f"unsupported intrinsics shape: {array.shape}")


def _photometric_consistency_from_psnr(psnr_db: float) -> float:
    return float(1.0 - 10.0 ** (-psnr_db / 20.0))


def _build_point_cloud(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    frames_resized: np.ndarray,
    *,
    device: str,
    max_pts_per_frame: int,
    conf: np.ndarray | None = None,
    conf_threshold: float = 0.5,
) -> tuple[Any, Any, Any] | tuple[None, None, None]:
    import torch

    n_frames, height, width = depth.shape
    all_pts: list[Any] = []
    all_colors: list[Any] = []
    all_ids: list[Any] = []

    for index in range(n_frames):
        depth_map = torch.from_numpy(depth[index]).float().to(device)
        intrinsics_i = torch.from_numpy(intrinsics[index]).double().to(device)
        extrinsics_i = torch.from_numpy(extrinsics[index]).double().to(device)
        rotation = extrinsics_i[:3, :3]
        translation = extrinsics_i[:3, 3]

        mask = depth_map > 1e-6
        if conf is not None:
            confidence = torch.from_numpy(conf[index]).float().to(device)
            mask = mask & (confidence > conf_threshold)
        if int(mask.sum()) == 0:
            continue

        ys, xs = torch.where(mask)
        depths = depth_map[mask].double()
        if len(depths) > max_pts_per_frame:
            pick = torch.randperm(len(depths), device=device)[:max_pts_per_frame]
            xs = xs[pick]
            ys = ys[pick]
            depths = depths[pick]

        pixels_h = torch.stack(
            [
                xs.double(),
                ys.double(),
                torch.ones_like(xs, dtype=torch.float64, device=device),
            ],
            dim=0,
        )
        camera_points = depths.unsqueeze(0) * (torch.linalg.inv(intrinsics_i) @ pixels_h)
        world_points = rotation.T @ (camera_points - translation.unsqueeze(1))

        all_pts.append(world_points.T.float())
        colors = torch.from_numpy(frames_resized[index]).float().to(device)
        all_colors.append(colors[ys.long(), xs.long()])
        all_ids.append(torch.full((world_points.shape[1],), index, dtype=torch.long, device=device))

    if not all_pts:
        return None, None, None
    return torch.cat(all_pts), torch.cat(all_colors), torch.cat(all_ids)


def _compute_wbench_metrics(
    world_pts: Any,
    colors: Any,
    source_ids: Any,
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    frames_resized: np.ndarray,
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    n_frames, height, width = depth.shape
    depth_tensor = torch.from_numpy(depth).float().to(device)
    frames_tensor = torch.from_numpy(frames_resized).float().to(device)
    extrinsics_tensor = torch.from_numpy(extrinsics).double().to(device)
    intrinsics_tensor = torch.from_numpy(intrinsics).double().to(device)
    points = world_pts.double().to(device)
    frame_masks = [source_ids == index for index in range(n_frames)]

    geo_errors: list[float] = []
    photo_psnrs: list[float] = []

    for index in range(n_frames):
        other_mask = ~frame_masks[index]
        points_other = points[other_mask]
        colors_other = colors[other_mask]
        if points_other.shape[0] == 0:
            geo_errors.append(float("inf"))
            photo_psnrs.append(0.0)
            continue

        rotation = extrinsics_tensor[index, :3, :3]
        translation = extrinsics_tensor[index, :3, 3]
        intrinsics_i = intrinsics_tensor[index]
        camera = rotation @ points_other.T + translation.unsqueeze(1)
        front = camera[2, :] > 1e-6
        camera_front = camera[:, front]
        colors_front = colors_other[front]
        if camera_front.shape[1] == 0:
            geo_errors.append(float("inf"))
            photo_psnrs.append(0.0)
            continue

        projected = intrinsics_i @ camera_front
        u = (projected[0] / projected[2]).float()
        v = (projected[1] / projected[2]).float()
        z = camera_front[2].float()

        valid = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = z[valid]
        colors_valid = colors_front[valid]
        if u_valid.shape[0] == 0:
            geo_errors.append(float("inf"))
            photo_psnrs.append(0.0)
            continue

        u_int = u_valid.round().long().clamp(0, width - 1)
        v_int = v_valid.round().long().clamp(0, height - 1)

        predicted_depth = depth_tensor[index, v_int, u_int]
        depth_valid = predicted_depth > 1e-6
        if int(depth_valid.sum()) > 0:
            rel_err = (z_valid[depth_valid] - predicted_depth[depth_valid]).abs() / predicted_depth[depth_valid]
            geo_errors.append(float(rel_err.median()))
        else:
            geo_errors.append(float("inf"))

        sort_idx = z_valid.argsort(descending=True)
        rendered = torch.zeros(height, width, 3, device=device)
        z_buffer = torch.full((height, width), float("inf"), device=device)
        rendered[v_int[sort_idx], u_int[sort_idx]] = colors_valid[sort_idx].float()
        z_buffer[v_int[sort_idx], u_int[sort_idx]] = z_valid[sort_idx]

        has_render = z_buffer < float("inf")
        if int(has_render.sum()) < 100:
            photo_psnrs.append(0.0)
            continue

        diff = (rendered[has_render] - frames_tensor[index][has_render]) / 255.0
        mse = float((diff**2).mean())
        photo_psnrs.append(float(-10.0 * np.log10(mse + 1e-10)))

    geo_array = np.array(geo_errors)
    valid_geo = geo_array[np.isfinite(geo_array)]
    if len(valid_geo) > 0:
        geo_result = {
            "geometric_consistency": float(1.0 / (1.0 + float(np.mean(valid_geo)))),
            "geo_mean_rel_error": float(np.mean(valid_geo)),
            "valid_frames": int(np.isfinite(geo_array).sum()),
        }
    else:
        geo_result = {"geometric_consistency": 0.0, "error": "all frames invalid"}

    psnr_array = np.array(photo_psnrs)
    valid_psnr = psnr_array[psnr_array > 0]
    photo_mean_psnr = float(np.mean(valid_psnr)) if len(valid_psnr) > 0 else 0.0
    photo_result = {
        "photo_mean_psnr": photo_mean_psnr,
        "photometric_consistency": _photometric_consistency_from_psnr(photo_mean_psnr) if photo_mean_psnr > 0 else 0.0,
        "valid_frames": int((psnr_array > 0).sum()),
    }
    return geo_result, photo_result


def _prepare_da3_geometry(
    prediction: Any,
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    depth_maps = np.asarray(prediction.depth, dtype=np.float32)
    if depth_maps.ndim != 3:
        raise ValueError(f"expected prediction depth with shape (T, H, W), got {depth_maps.shape}")

    extrinsics = _as_w2c_extrinsics(np.asarray(prediction.extrinsics))
    intrinsics = _as_intrinsics_33(np.asarray(prediction.intrinsics))
    confidence = None
    if getattr(prediction, "conf", None) is not None:
        confidence = np.asarray(prediction.conf, dtype=np.float32)

    aligned_frames = np.stack(
        [
            _resize_rgb(frame, depth_maps[index].shape)
            for index, frame in enumerate(frames[: depth_maps.shape[0]])
        ],
        axis=0,
    ).astype(np.uint8)
    return aligned_frames, depth_maps, extrinsics, intrinsics, confidence


def compute_reconstruction_consistency(
    frames: Sequence[np.ndarray],
    *,
    prompt: str | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute WBench-aligned reconstruction consistency from DA3 depth/pose."""
    payload = dict(runtime or {})
    backend = DEFAULT_RECONSTRUCTION_BACKEND
    uint8_frames = _to_uint8_frames(frames)
    if len(uint8_frames) < 3:
        return {
            "raw": None,
            "backend": backend,
            "details": {"frame_count": len(uint8_frames)},
            "error": "reconstruction_consistency requires at least three prediction frames",
        }

    _ensure_cuda_toolkit_env()
    device = str(payload.get("device") or _default_device())
    source = _resolve_depth_anything3_source(payload.get("repo_root"))
    model_name = _resolve_model_name(str(payload.get("model_name") or DEFAULT_RECONSTRUCTION_MODEL))
    process_res = int(payload.get("process_res", 504))
    max_pts_per_frame = int(payload.get("max_pts_per_frame", DEFAULT_MAX_PTS_PER_FRAME))
    conf_threshold = float(payload.get("conf_threshold", 0.5))
    ref_view_strategy = str(payload.get("ref_view_strategy") or "saddle_balanced")
    use_ray_pose = bool(payload.get("use_ray_pose", False))
    use_confidence = bool(payload.get("use_confidence_mask", False))

    model = _load_depth_anything3_model(
        source_root=str(source.root),
        package_name=source.package_name,
        model_name=model_name,
        device=device,
    )

    import torch

    with torch.inference_mode():
        prediction = model.inference(
            image=uint8_frames,
            extrinsics=None,
            intrinsics=None,
            process_res=process_res,
            align_to_input_ext_scale=True,
            infer_gs=False,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            export_dir=None,
        )

    if getattr(prediction, "depth", None) is None:
        return {
            "raw": None,
            "backend": backend,
            "details": {"frame_count": len(uint8_frames)},
            "error": "Depth-Anything-3 did not return depth maps for reconstruction consistency",
        }

    frames_resized, depth_maps, extrinsics, intrinsics, confidence = _prepare_da3_geometry(
        prediction,
        uint8_frames,
    )
    world_pts, colors, source_ids = _build_point_cloud(
        depth_maps,
        extrinsics,
        intrinsics,
        frames_resized,
        device=device,
        max_pts_per_frame=max_pts_per_frame,
        conf=confidence if use_confidence else None,
        conf_threshold=conf_threshold,
    )
    if world_pts is None:
        return {
            "raw": None,
            "backend": backend,
            "details": {"frame_count": len(uint8_frames)},
            "error": "point cloud construction failed for reconstruction consistency",
        }

    geo_result, photo_result = _compute_wbench_metrics(
        world_pts,
        colors,
        source_ids,
        depth_maps,
        extrinsics,
        intrinsics,
        frames_resized,
        device=device,
    )
    geometric = float(geo_result.get("geometric_consistency", 0.0))
    photo_psnr = float(photo_result.get("photo_mean_psnr", 0.0))
    photometric = float(photo_result.get("photometric_consistency", 0.0))
    details = {
        "source_metric": "reconstruction_consistency",
        "method": "wbench_da3_depth_reprojection",
        "prompt": prompt,
        "frame_count": len(uint8_frames),
        "num_frames": int(depth_maps.shape[0]),
        "process_res": process_res,
        "model_name": model_name,
        "source_root": str(source.root),
        "package_name": source.package_name,
        "device": device,
        "max_pts_per_frame": max_pts_per_frame,
        # Keep full precision internally. WBench rounds the per-video geometric
        # score to 4 decimals and PSNR to 2 before reporting; WorldArena only
        # rounds when serializing aggregate summaries.
        "geometric_consistency": geometric,
        "geo_mean_rel_error": float(geo_result.get("geo_mean_rel_error", 0.0)),
        "photometric_psnr": photo_psnr,
        "photometric_consistency": photometric,
        "geo_valid_frames": int(geo_result.get("valid_frames", 0)),
        "photo_valid_frames": int(photo_result.get("valid_frames", 0)),
    }
    if geo_result.get("error"):
        details["geo_error"] = geo_result["error"]
    return {
        "raw": geometric,
        "backend": backend,
        "details": details,
        "error": None if geometric > 0.0 or photo_psnr > 0.0 else "reconstruction consistency unavailable",
    }


__all__ = [
    "DEFAULT_RECONSTRUCTION_BACKEND",
    "DEFAULT_RECONSTRUCTION_FPS",
    "DEFAULT_RECONSTRUCTION_MODEL",
    "DEFAULT_MAX_PTS_PER_FRAME",
    "_compute_wbench_metrics",
    "_photometric_consistency_from_psnr",
    "compute_reconstruction_consistency",
    "extract_wbench_frames",
]
