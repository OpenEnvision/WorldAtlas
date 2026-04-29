from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import traceback
from typing import Any

import cv2
import numpy as np
import torch

from worldarena.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence
from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldarena.models.adapters.worldfm_runner import (
    _intrinsics_vector_to_matrix,
    _resolve_checkpoint_path,
    _resolve_model_reference,
    _resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena persistent WorldFM batch runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--hw_path", default=None, type=str)
    parser.add_argument("--moge_path", default=None, type=str)
    parser.add_argument("--moge_pretrained", default=None, type=str)
    parser.add_argument("--model_filename", default="worldfm_2-step.pth", type=str)
    parser.add_argument("--vae_subdir", default="vae", type=str)
    parser.add_argument("--step", default=2, type=int)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--cfg_scale", default=4.5, type=float)
    parser.add_argument("--render_size", default=512, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--video_crf", default=14, type=int)
    parser.add_argument("--video_preset", default="medium", type=str)
    parser.add_argument("--cache_condition_latents", action="store_true")
    return parser.parse_args()


def _load_official_pipeline(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location(
        "worldarena_worldfm_official_run_pipeline",
        repo_root / "run_pipeline.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load WorldFM run_pipeline.py from {repo_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            for key in ("sample_id", "prediction_stem", "conditioning_image", "output_path", "annotation_path"):
                if key not in payload:
                    raise KeyError(f"{path}:{line_number} missing required key: {key}")
            requests.append(payload)
    return requests


def _image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def _k_already_in_render_space(k_matrix: np.ndarray, render_size: int) -> bool:
    fx, fy = float(k_matrix[0, 0]), float(k_matrix[1, 1])
    cx, cy = float(k_matrix[0, 2]), float(k_matrix[1, 2])
    size = float(render_size)
    return (
        0.0 <= cx <= size
        and 0.0 <= cy <= size
        and 0.0 < fx <= size * 4.0
        and 0.0 < fy <= size * 4.0
    )


def _synthetic_k_for_render(k_matrix: np.ndarray, render_size: int) -> np.ndarray:
    source_width = max(float(k_matrix[0, 2]) * 2.0, 1.0)
    source_height = max(float(k_matrix[1, 2]) * 2.0, 1.0)
    source_max = max(source_width, source_height)
    focal_scale = 0.5 * (float(k_matrix[0, 0]) + float(k_matrix[1, 1])) / source_max
    focal = float(render_size) * focal_scale
    center = float(render_size) / 2.0
    return np.asarray([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float64)


def _scale_k_center_crop(k_matrix: np.ndarray, source_size: tuple[int, int], render_size: int) -> np.ndarray:
    width, height = source_size
    target = float(render_size)
    scale = max(target / max(float(width), 1.0), target / max(float(height), 1.0))
    resized_width = float(width) * scale
    resized_height = float(height) * scale
    crop_x = max(0.0, (resized_width - target) / 2.0)
    crop_y = max(0.0, (resized_height - target) / 2.0)
    out = np.asarray(k_matrix, dtype=np.float64).copy()
    out[0, 0] *= scale
    out[1, 1] *= scale
    out[0, 2] = out[0, 2] * scale - crop_x
    out[1, 2] = out[1, 2] * scale - crop_y
    return out


def _normalize_k_for_render(
    k_matrix: np.ndarray,
    *,
    render_size: int,
    conditioning_image: Path | None,
    pose_source: str | None,
) -> np.ndarray:
    render_size = int(render_size)
    if render_size <= 0 or _k_already_in_render_space(k_matrix, render_size):
        return np.asarray(k_matrix, dtype=np.float64)

    if str(pose_source or "").lower() == "synthetic":
        return _synthetic_k_for_render(k_matrix, render_size)

    if conditioning_image is not None:
        source_size = _image_size(conditioning_image)
        if source_size is not None:
            return _scale_k_center_crop(k_matrix, source_size, render_size)

    return np.asarray(k_matrix, dtype=np.float64)


def _camera_payload(
    annotation_path: str,
    *,
    render_size: int | None = None,
    conditioning_image: Path | None = None,
    pose_source: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    camera_matrices = load_camera_matrices(annotation_path)
    if camera_matrices is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")
    intrinsics = load_intrinsics_sequence(annotation_path, target_frames=len(camera_matrices))
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {annotation_path}")
    if len(intrinsics) > 1 and not np.allclose(intrinsics, intrinsics[0], atol=1e-4):
        print(
            "[WorldArena][WorldFM] intrinsics vary across frames; using the first matrix for all poses.",
            flush=True,
        )
    k_matrix = np.asarray(_intrinsics_vector_to_matrix(intrinsics[0]), dtype=np.float64)
    if render_size is not None:
        k_matrix = _normalize_k_for_render(
            k_matrix,
            render_size=int(render_size),
            conditioning_image=conditioning_image,
            pose_source=pose_source,
        )
    c2w_list = [np.asarray(item, dtype=np.float64) for item in camera_matrices]
    return k_matrix, c2w_list


def _official_config(
    pipeline,
    *,
    config_path: Path | None,
    hw_path: Path | None,
    moge_path: Path | None,
    moge_pretrained: str | None,
    model_path: Path,
    vae_path: Path,
    args: argparse.Namespace,
):
    namespace = SimpleNamespace(
        config=str(config_path) if config_path is not None else "",
        output_dir="outputs",
        hw_path=str(hw_path) if hw_path is not None else "",
        moge_path=str(moge_path) if moge_path is not None else "",
        moge_pretrained=moge_pretrained,
        render_size=int(args.render_size),
        model_path=str(model_path),
        vae_path=str(vae_path),
        image_size=int(args.image_size),
        step=int(args.step),
        cfg_scale=float(args.cfg_scale),
        gpu_index=int(args.gpu_index),
    )
    return pipeline._load_config(namespace)


def _enable_panorama_matrix_cache(pipeline) -> None:
    merge_fn = pipeline.moge_pano.merge_panorama_depth
    globals_map = getattr(merge_fn, "__globals__", {})
    for name in ("grad_equation", "poisson_equation"):
        fn = globals_map.get(name)
        if fn is None or hasattr(fn, "cache_info"):
            continue
        globals_map[name] = lru_cache(maxsize=8)(fn)


def _make_panogen_demo(pipeline, cfg):
    class _Args:
        fp8_attention = bool(cfg.panogen.fp8_attention)
        fp8_gemm = bool(cfg.panogen.fp8_gemm)
        cache = bool(cfg.panogen.cache)

    return pipeline.Image2PanoramaDemo(_Args())


class _CachedMoGeProvider:
    def __init__(self, model) -> None:
        self._model = model

    def from_pretrained(self, _pretrained: str):
        return self._model


class _Runtime:
    def __init__(self, *, pipeline, cfg) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        self.panogen_demo = _make_panogen_demo(pipeline, cfg)
        self.moge_model = pipeline.moge_pano.MoGeModel.from_pretrained(str(cfg.moge.pretrained))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moge_model = self.moge_model.to(device).eval()
        self.worldfm_svc, self.worldfm_cfg = pipeline.step4_init(cfg=cfg)

    def step1_panorama(self, image_path: Path, *, prompt: str, negative_prompt: str = ""):
        return self.panogen_demo.run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=str(image_path),
            seed=int(self.cfg.panogen.seed),
            save_to_disk=False,
            output_path=None,
        )

    def step2_moge(self, panorama_img, output_dir: Path):
        original = self.pipeline.moge_pano.MoGeModel
        self.pipeline.moge_pano.MoGeModel = _CachedMoGeProvider(self.moge_model)
        try:
            return self.pipeline.step2_moge_pipeline(
                panorama_img=panorama_img,
                output_dir=output_dir,
                cfg=self.cfg,
                pretrained=str(self.cfg.moge.pretrained),
            )
        finally:
            self.pipeline.moge_pano.MoGeModel = original


def _render_one_with_condition_index(
    runtime: _Runtime,
    renderer,
    cond_db,
    pp_result,
    k_matrix: np.ndarray,
    c2w: np.ndarray,
    *,
    rcfg,
    render_size: int,
):
    out = renderer.render_torch(
        K_3x3=np.asarray(k_matrix, dtype=np.float64),
        c2w_4x4=np.asarray(c2w, dtype=np.float64),
        c2w_is_camera_to_world=True,
    )
    idx, hits, samples = runtime.pipeline.select_best_condition_index(
        depth_cur=out.depth_f32,
        K_cur=np.asarray(k_matrix, dtype=np.float64),
        c2w_cur=np.asarray(c2w, dtype=np.float64),
        cond_db=cond_db,
        sample_grid=int(rcfg.sample_grid),
        center_grid=int(rcfg.center_grid),
        center_frac=float(rcfg.center_frac),
        eps_rel=float(rcfg.eps_rel),
        eps_abs=float(rcfg.eps_abs),
        px_radius=int(rcfg.px_radius),
        max_view_angle_deg=float(rcfg.max_view_angle_deg),
        use_distance_weight=bool(rcfg.use_distance_weight),
        dist_min_m=float(rcfg.dist_min_m),
        dist_max_m=float(rcfg.dist_max_m),
        weight_near=float(rcfg.weight_near),
        weight_far=float(rcfg.weight_far),
    )
    return out.rgb_u8, int(idx), int(hits), int(samples)


def _cache_condition_candidates(runtime: _Runtime, pp_result, work_dir: Path) -> None:
    condition_dir = work_dir / "condition_candidates"
    condition_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, image_rgb in enumerate(pp_result.condition_images):
        path = condition_dir / f"{index:04d}.png"
        if not path.exists():
            cv2.imwrite(str(path), image_rgb[:, :, ::-1])
        paths.append(str(path))
    runtime.worldfm_svc.set_cond2_candidates_from_paths(paths, chunk=8)


def _infer_worldfm_frame(
    runtime: _Runtime,
    render_u8,
    *,
    cond2_rgb: np.ndarray | None = None,
    cond2_index: int | None = None,
) -> np.ndarray:
    if cond2_index is None:
        if cond2_rgb is None:
            raise ValueError("cond2_rgb is required when condition latent cache is disabled")
        runtime.worldfm_svc.set_cond2_from_array(np.asarray(cond2_rgb, dtype=np.uint8))
        decoded = runtime.worldfm_svc.infer_from_render_u8(render_u8)
    else:
        decoded = runtime.worldfm_svc.infer_from_render_u8(render_u8, cond2_index=int(cond2_index))
    return (
        torch.clamp(127.5 * decoded[0] + 128.0, 0, 255)
        .permute(1, 2, 0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _write_video_rgb_frames(
    output_path: Path,
    frames: list[np.ndarray],
    *,
    fps: int,
    crf: int,
    preset: str,
) -> None:
    if not frames:
        raise RuntimeError(f"cannot write empty video: {output_path}")
    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.shape[:2] != (height, width) or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"inconsistent frame shape for {output_path}: {rgb.shape}")
            process.stdin.write(np.ascontiguousarray(rgb).tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code} while writing {output_path}")


def _write_prediction_video(
    runtime: _Runtime,
    *,
    request: dict[str, Any],
    output_path: Path,
    work_dir: Path,
    fps: int,
    video_crf: int,
    video_preset: str,
    cache_condition_latents: bool,
) -> None:
    image_path = Path(str(request["conditioning_image"])).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"conditioning image not found: {image_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    tmp_path.unlink(missing_ok=True)
    frames: list[np.ndarray] = []
    renderer = cond_db = pp_result = panorama_img = None
    try:
        panorama_prompt = " ".join(str(request.get("panorama_prompt") or request.get("prompt") or "").split())
        if panorama_prompt:
            print(f"[WorldArena][WorldFM] Panorama prompt: {panorama_prompt}", flush=True)
        panorama_negative_prompt = " ".join(str(request.get("panorama_negative_prompt") or "").split())
        if panorama_negative_prompt:
            print(f"[WorldArena][WorldFM] Panorama negative prompt: {panorama_negative_prompt}", flush=True)
        panorama_img = runtime.step1_panorama(
            image_path,
            prompt=panorama_prompt,
            negative_prompt=panorama_negative_prompt,
        )
        pp_result = runtime.step2_moge(panorama_img, work_dir)
        print("[WorldArena][WorldFM] Initializing renderer and condition DB", flush=True)
        renderer, cond_db, rcfg, size = runtime.pipeline.step3_init(pp_result, cfg=runtime.cfg)
        k_matrix, c2w_list = _camera_payload(
            str(request["annotation_path"]),
            render_size=int(size),
            conditioning_image=image_path,
            pose_source=str(request.get("pose_source") or ""),
        )
        if cache_condition_latents:
            _cache_condition_candidates(runtime, pp_result, work_dir)
        for frame_index, c2w in enumerate(c2w_list, start=1):
            print(
                f"[WorldArena][WorldFM] {request['prediction_stem']} frame {frame_index}/{len(c2w_list)}",
                flush=True,
            )
            render_u8, cond2_index, hits, samples = _render_one_with_condition_index(
                runtime,
                renderer,
                cond_db,
                pp_result,
                k_matrix,
                c2w,
                rcfg=rcfg,
                render_size=size,
            )
            print(
                f"[WorldArena][WorldFM] Selected condition: idx={cond2_index} hits={hits}/{samples}",
                flush=True,
            )
            if cache_condition_latents:
                frame = _infer_worldfm_frame(runtime, render_u8, cond2_index=cond2_index)
            else:
                frame = _infer_worldfm_frame(
                    runtime,
                    render_u8,
                    cond2_rgb=pp_result.condition_images[int(cond2_index)],
                )
            frames.append(frame)
        _write_video_rgb_frames(
            tmp_path,
            frames,
            fps=int(fps),
            crf=int(video_crf),
            preset=str(video_preset),
        )
        output_path.unlink(missing_ok=True)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists() and not output_path.exists():
            tmp_path.unlink(missing_ok=True)
        del renderer, cond_db, pp_result, panorama_img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("WorldFM batch runner requires checkpoint_dir")
    config_path = _resolve_repo_path(repo_root, args.config_path)
    hw_path = _resolve_repo_path(repo_root, args.hw_path)
    moge_path = _resolve_repo_path(repo_root, args.moge_path)
    model_path = _resolve_checkpoint_path(checkpoint_dir, args.model_filename)
    vae_path = _resolve_checkpoint_path(checkpoint_dir, args.vae_subdir)
    moge_pretrained = _resolve_model_reference(args.moge_pretrained)

    os.environ.update(apply_checkpoint_env())
    os.chdir(repo_root)
    pipeline = _load_official_pipeline(repo_root)
    cfg = _official_config(
        pipeline,
        config_path=config_path,
        hw_path=hw_path,
        moge_path=moge_path,
        moge_pretrained=moge_pretrained,
        model_path=model_path,
        vae_path=vae_path,
        args=args,
    )
    if cfg.pipeline.gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(int(cfg.pipeline.gpu_index))

    pipeline.setup_external_repos(
        hw_path=str(cfg.submodules.hw_path),
        moge_path=str(cfg.submodules.moge_path),
    )
    _enable_panorama_matrix_cache(pipeline)
    requests = _load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    print(f"[WorldArena][WorldFM] Loaded {len(requests)} requests", flush=True)
    runtime = _Runtime(pipeline=pipeline, cfg=cfg)
    results_path = Path(args.batch_spec_path).with_suffix(".results.jsonl")
    results_path.unlink(missing_ok=True)
    work_root = Path(args.batch_spec_path).with_suffix(".work")
    work_root.mkdir(parents=True, exist_ok=True)

    with results_path.open("a", encoding="utf-8") as results_file:
        for index, request in enumerate(requests, start=1):
            output_path = Path(str(request["output_path"])).expanduser().resolve()
            print(
                f"[WorldArena][WorldFM] Batch item {index}/{len(requests)}: {request['prediction_stem']}",
                flush=True,
            )
            record: dict[str, Any] = {
                "sample_id": request["sample_id"],
                "prediction_path": str(output_path),
                "status": "pending",
                "error": None,
            }
            try:
                _write_prediction_video(
                    runtime,
                    request=request,
                    output_path=output_path,
                    work_dir=work_root / str(request["prediction_stem"]),
                    fps=int(args.fps),
                    video_crf=int(args.video_crf),
                    video_preset=str(args.video_preset),
                    cache_condition_latents=bool(args.cache_condition_latents),
                )
                record["status"] = "generated"
            except Exception as exc:  # noqa: BLE001 - keep later samples running.
                record["status"] = "failed"
                record["error"] = str(exc)
                print(
                    f"[WorldArena][WorldFM] FAILED {request['prediction_stem']}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()


if __name__ == "__main__":
    main()
