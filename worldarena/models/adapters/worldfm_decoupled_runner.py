from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
import torch
from tqdm import trange

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldarena.models.adapters.worldfm_batch_runner import (
    _camera_payload,
    _enable_panorama_matrix_cache,
    _infer_worldfm_frame,
    _load_batch_spec,
    _load_official_pipeline,
    _official_config,
    _render_one_with_condition_index,
    _resolve_checkpoint_path,
    _resolve_model_reference,
    _resolve_repo_path,
    _write_video_rgb_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena decoupled WorldFM stage runner.")
    parser.add_argument("--stage", required=True, choices=["prepare", "merge", "final"])
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--work_root", required=True, type=str)
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
    parser.add_argument("--worker_name", default="", type=str)
    parser.add_argument("--poll_seconds", default=30.0, type=float)
    parser.add_argument("--lock_timeout_seconds", default=21600.0, type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete_after_final", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stage_dir(work_root: Path, stage: str, request: dict[str, Any]) -> Path:
    return work_root / stage / str(request["prediction_stem"])


def _output_root_from_request(request: dict[str, Any]) -> Path:
    return Path(str(request["output_path"])).expanduser().resolve().parent


def _publish_stage_staging(
    work_root: Path,
    request: dict[str, Any],
    stage: str,
    stage_dir: Path,
) -> None:
    output_root = _output_root_from_request(request)
    stem = str(request["prediction_stem"])
    dest = output_root / "_staging" / stem / stage
    dest.mkdir(parents=True, exist_ok=True)
    if stage == "prepare":
        for name in ("panorama_rgb.png", "moge_prepare.npz"):
            source = stage_dir / name
            if source.exists():
                shutil.copy2(source, dest / name)
    elif stage == "merge":
        for name in ("transforms_condition.json",):
            source = stage_dir / name
            if source.exists():
                shutil.copy2(source, dest / name)
        preview = stage_dir / "condition_candidates" / "0000.png"
        if preview.exists():
            shutil.copy2(preview, dest / "condition_preview.png")
    done_path = _done_path(work_root, stage, request)
    if done_path.exists():
        shutil.copy2(done_path, dest / "done.json")


def _done_path(work_root: Path, stage: str, request: dict[str, Any]) -> Path:
    return _stage_dir(work_root, stage, request) / "done.json"


def _fail_path(work_root: Path, stage: str, request: dict[str, Any]) -> Path:
    return _stage_dir(work_root, stage, request) / "failed.json"


def _lock_path(work_root: Path, stage: str, request: dict[str, Any]) -> Path:
    return _stage_dir(work_root, stage, request) / f"{stage}.lock"


def _is_stale(path: Path, *, timeout_seconds: float) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > float(timeout_seconds)


def _claim(path: Path, *, timeout_seconds: float) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and _is_stale(path, timeout_seconds=timeout_seconds):
        path.unlink(missing_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(json.dumps({"pid": os.getpid(), "time": time.time()}, ensure_ascii=False))
    return True


def _release(path: Path) -> None:
    path.unlink(missing_ok=True)


def _ensure_utils3d_depth_map_to_point_map() -> None:
    import utils3d.torch

    if hasattr(utils3d.torch, "depth_map_to_point_map"):
        return

    def depth_map_to_point_map(
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return utils3d.torch.depth_to_points(depth, intrinsics=intrinsics, extrinsics=extrinsics)

    utils3d.torch.depth_map_to_point_map = depth_map_to_point_map


def _complete(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    _write_json(tmp_path, payload)
    tmp_path.replace(path)


def _failed(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    _write_json(tmp_path, payload)
    tmp_path.replace(path)


def _pipeline_context(args: argparse.Namespace, *, require_gpu: bool):
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("WorldFM stage runner requires checkpoint_dir")
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
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{args.stage} stage requires CUDA")
        if cfg.pipeline.gpu_index >= 0:
            torch.cuda.set_device(int(cfg.pipeline.gpu_index))
    return repo_root, pipeline, cfg


def _make_panogen_demo(pipeline, cfg):
    class _Args:
        fp8_attention = bool(cfg.panogen.fp8_attention)
        fp8_gemm = bool(cfg.panogen.fp8_gemm)
        cache = bool(cfg.panogen.cache)

    return pipeline.Image2PanoramaDemo(_Args())


class _PrepareRuntime:
    def __init__(self, *, pipeline, cfg) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        pipeline.setup_external_repos(
            hw_path=str(cfg.submodules.hw_path),
            moge_path=str(cfg.submodules.moge_path),
        )
        self.panogen_demo = _make_panogen_demo(pipeline, cfg)
        self.moge_model = pipeline.moge_pano.MoGeModel.from_pretrained(str(cfg.moge.pretrained))
        self.moge_model = self.moge_model.to(torch.device("cuda")).eval()

    def close(self) -> None:
        del self.moge_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _FinalRuntime:
    def __init__(self, *, pipeline, cfg) -> None:
        self.pipeline = pipeline
        self.cfg = cfg
        self.worldfm_svc, self.worldfm_cfg = pipeline.step4_init(cfg=cfg)


def _prepare_one(runtime: _PrepareRuntime, request: dict[str, Any], stage_dir: Path) -> None:
    image_path = Path(str(request["conditioning_image"])).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"conditioning image not found: {image_path}")

    prompt = " ".join(str(request.get("panorama_prompt") or request.get("prompt") or "").split())
    print(f"[WorldArena][WorldFM][prepare] {request['prediction_stem']} prompt={prompt}", flush=True)
    negative_prompt = " ".join(str(request.get("panorama_negative_prompt") or "").split())
    if negative_prompt:
        print(
            f"[WorldArena][WorldFM][prepare] {request['prediction_stem']} negative_prompt={negative_prompt}",
            flush=True,
        )
    panorama_img = runtime.panogen_demo.run(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path=str(image_path),
        seed=int(runtime.cfg.panogen.seed),
        save_to_disk=False,
        output_path=None,
    )

    image_rgb = np.array(panorama_img)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    tier = runtime.pipeline.select_tier(orig_w)
    width, height = int(tier["width"]), int(tier["height"])
    split_resolution = int(tier["split_res"])
    if orig_w != width or orig_h != height:
        interp = cv2.INTER_AREA if width < orig_w else cv2.INTER_LINEAR
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=interp)

    import utils3d

    _ensure_utils3d_depth_map_to_point_map()
    mcfg = runtime.cfg.moge
    extrinsics, intrinsics = runtime.pipeline._get_panorama_cameras(
        int(mcfg.num_views),
        float(mcfg.fov_deg),
    )
    splitted_images = runtime.pipeline.moge_pano.split_panorama_image(
        image_rgb,
        extrinsics,
        intrinsics,
        split_resolution,
    )
    splitted_dist, splitted_masks = [], []
    batch_size = int(mcfg.batch_size)
    device = torch.device("cuda")
    for index in trange(0, len(splitted_images), batch_size, desc="MoGe Infer", leave=False):
        batch = np.stack(splitted_images[index:index + batch_size])
        tensor = torch.tensor(batch / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        fov_x, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(intrinsics[index:index + batch_size])))
        fov_x_t = torch.tensor(fov_x, dtype=torch.float32, device=device)
        out = runtime.moge_model.infer(
            tensor,
            resolution_level=int(mcfg.resolution_level),
            fov_x=fov_x_t,
            apply_mask=False,
        )
        splitted_dist.extend(list(out["points"].norm(dim=-1).cpu().numpy()))
        splitted_masks.extend(list(out["mask"].cpu().numpy()))

    stage_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stage_dir / "panorama_rgb.png"), image_rgb[:, :, ::-1])
    np.savez(
        stage_dir / "moge_prepare.npz",
        splitted_dist=np.stack(splitted_dist).astype(np.float32),
        splitted_masks=np.stack(splitted_masks),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        width=np.asarray(width, dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        merge_width=np.asarray(min(int(mcfg.merge_max_width), width), dtype=np.int32),
        merge_height=np.asarray(min(int(mcfg.merge_max_height), height), dtype=np.int32),
    )


def _init_merge_pipeline(repo_root: Path, cfg):
    pipeline = _load_official_pipeline(repo_root)
    resolved_moge = str(cfg.submodules.moge_path) or str(repo_root / "submodules" / "MoGe")
    pipeline.ensure_moge(resolved_moge)
    _enable_panorama_matrix_cache(pipeline)
    return pipeline


def _merge_one(pipeline, work_root: Path, request: dict[str, Any], stage_dir: Path) -> None:
    prepare_dir = _stage_dir(work_root, "prepare", request)
    prep_path = prepare_dir / "moge_prepare.npz"
    panorama_path = prepare_dir / "panorama_rgb.png"
    if not prep_path.exists() or not panorama_path.exists():
        raise FileNotFoundError(f"missing prepare artifact for {request['prediction_stem']}")

    payload = np.load(prep_path)
    image_bgr = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to load panorama image: {panorama_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    width, height = int(payload["width"]), int(payload["height"])
    merge_w, merge_h = int(payload["merge_width"]), int(payload["merge_height"])
    splitted_dist = [item for item in payload["splitted_dist"]]
    splitted_masks = [item for item in payload["splitted_masks"]]
    extrinsics = [item for item in payload["extrinsics"]]
    intrinsics = [item for item in payload["intrinsics"]]

    print(
        f"[WorldArena][WorldFM][merge] {request['prediction_stem']} merge={merge_w}x{merge_h}",
        flush=True,
    )
    panorama_depth, panorama_mask = pipeline.moge_pano.merge_panorama_depth(
        merge_w,
        merge_h,
        splitted_dist,
        splitted_masks,
        extrinsics,
        intrinsics,
    )
    panorama_depth = panorama_depth.astype(np.float32)
    panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
    depth_raw = panorama_depth.copy()
    if panorama_mask.any():
        depth_raw[~panorama_mask] = panorama_depth[panorama_mask].max()
    depth_raw = depth_raw / 100.0

    result = pipeline.postprocess_panorama(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        depth_raw,
        save_dir=None,
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.save(stage_dir / "ply_xyz.npy", result.ply_xyz)
    np.save(stage_dir / "ply_rgb.npy", result.ply_rgb)
    condition_dir = stage_dir / "condition_candidates"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(result.condition_images):
        cv2.imwrite(str(condition_dir / f"{index:04d}.png"), image[:, :, ::-1])
    _write_json(stage_dir / "transforms_condition.json", result.transforms)


def _load_postprocess_result(pipeline, stage_dir: Path):
    condition_dir = stage_dir / "condition_candidates"
    condition_images: list[np.ndarray] = []
    for path in sorted(condition_dir.glob("*.png")):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to load condition image: {path}")
        condition_images.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not condition_images:
        raise RuntimeError(f"no condition images found under {condition_dir}")
    return SimpleNamespace(
        ply_xyz=np.load(stage_dir / "ply_xyz.npy"),
        ply_rgb=np.load(stage_dir / "ply_rgb.npy"),
        condition_images=condition_images,
        transforms=_read_json(stage_dir / "transforms_condition.json"),
    )


def _final_one(runtime: _FinalRuntime, work_root: Path, request: dict[str, Any], args: argparse.Namespace) -> None:
    output_path = Path(str(request["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    tmp_path.unlink(missing_ok=True)

    pp_result = _load_postprocess_result(runtime.pipeline, _stage_dir(work_root, "merge", request))
    renderer = cond_db = None
    try:
        print(f"[WorldArena][WorldFM][final] init {request['prediction_stem']}", flush=True)
        renderer, cond_db, rcfg, size = runtime.pipeline.step3_init(pp_result, cfg=runtime.cfg)
        k_matrix, c2w_list = _camera_payload(
            str(request["annotation_path"]),
            render_size=int(size),
            conditioning_image=Path(str(request["conditioning_image"])).expanduser().resolve(),
            pose_source=str(request.get("pose_source") or ""),
        )
        frames: list[np.ndarray] = []
        for frame_index, c2w in enumerate(c2w_list, start=1):
            print(
                f"[WorldArena][WorldFM][final] {request['prediction_stem']} frame {frame_index}/{len(c2w_list)}",
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
                f"[WorldArena][WorldFM][final] Selected condition: idx={cond2_index} hits={hits}/{samples}",
                flush=True,
            )
            frame = _infer_worldfm_frame(
                runtime,
                render_u8,
                cond2_rgb=pp_result.condition_images[int(cond2_index)],
            )
            frames.append(frame)
        _write_video_rgb_frames(
            tmp_path,
            frames,
            fps=int(args.fps),
            crf=int(args.video_crf),
            preset=str(args.video_preset),
        )
        output_path.unlink(missing_ok=True)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists() and not output_path.exists():
            tmp_path.unlink(missing_ok=True)
        del renderer, cond_db, pp_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _stage_done(args: argparse.Namespace, work_root: Path, request: dict[str, Any]) -> bool:
    output_path = Path(str(request["output_path"])).expanduser().resolve()
    if args.stage == "final" and output_path.exists() and not args.overwrite:
        done = _done_path(work_root, "final", request)
        if not done.exists():
            _complete(done, {"sample_id": request["sample_id"], "status": "generated", "output_path": str(output_path)})
        return True
    return _done_path(work_root, args.stage, request).exists()


def _dependency_ready(work_root: Path, request: dict[str, Any], stage: str) -> bool:
    if stage == "prepare":
        return True
    if stage == "merge":
        return _done_path(work_root, "prepare", request).exists()
    if stage == "final":
        return _done_path(work_root, "merge", request).exists()
    raise ValueError(stage)


def _dependency_error(work_root: Path, request: dict[str, Any], stage: str) -> str | None:
    previous = {"merge": "prepare", "final": "merge"}.get(stage)
    if previous is None:
        return None
    path = _fail_path(work_root, previous, request)
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return f"{previous} stage failed"
    return str(payload.get("error") or f"{previous} stage failed")


def _run_stage_loop(
    *,
    args: argparse.Namespace,
    work_root: Path,
    requests: list[dict[str, Any]],
    process_one: Callable[[dict[str, Any]], None],
) -> None:
    worker = args.worker_name or f"{args.stage}-{os.getpid()}"
    while True:
        made_progress = False
        blocked = False
        for request in requests:
            if _stage_done(args, work_root, request):
                continue
            if _fail_path(work_root, args.stage, request).exists():
                continue
            dependency_error = _dependency_error(work_root, request, args.stage)
            if dependency_error:
                fail_path = _fail_path(work_root, args.stage, request)
                if not fail_path.exists():
                    _failed(
                        fail_path,
                        {
                            "sample_id": request.get("sample_id"),
                            "prediction_stem": request.get("prediction_stem"),
                            "status": "failed",
                            "worker": worker,
                            "error": f"dependency failed: {dependency_error}",
                            "time": time.time(),
                        },
                    )
                continue
            if not _dependency_ready(work_root, request, args.stage):
                blocked = True
                continue
            lock = _lock_path(work_root, args.stage, request)
            if not _claim(lock, timeout_seconds=float(args.lock_timeout_seconds)):
                blocked = True
                continue
            made_progress = True
            fail_path = _fail_path(work_root, args.stage, request)
            done_path = _done_path(work_root, args.stage, request)
            try:
                print(
                    f"[WorldArena][WorldFM][{args.stage}] worker={worker} sample={request['prediction_stem']}",
                    flush=True,
                )
                process_one(request)
                _complete(
                    done_path,
                    {
                        "sample_id": request["sample_id"],
                        "prediction_stem": request["prediction_stem"],
                        "status": "done",
                        "worker": worker,
                        "time": time.time(),
                    },
                )
                if args.stage in {"prepare", "merge"}:
                    _publish_stage_staging(
                        work_root,
                        request,
                        args.stage,
                        _stage_dir(work_root, args.stage, request),
                    )
            except Exception as exc:  # noqa: BLE001 - write per-sample failure and continue.
                _failed(
                    fail_path,
                    {
                        "sample_id": request.get("sample_id"),
                        "prediction_stem": request.get("prediction_stem"),
                        "status": "failed",
                        "worker": worker,
                        "error": str(exc),
                        "time": time.time(),
                    },
                )
                print(
                    f"[WorldArena][WorldFM][{args.stage}] FAILED {request.get('prediction_stem')}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
            finally:
                _release(lock)
        if not made_progress:
            if not blocked:
                break
            time.sleep(float(args.poll_seconds))


def _delete_intermediates(work_root: Path, request: dict[str, Any]) -> None:
    shutil.rmtree(_stage_dir(work_root, "prepare", request), ignore_errors=True)
    shutil.rmtree(_stage_dir(work_root, "merge", request), ignore_errors=True)


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    requests = _load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    if args.stage == "prepare":
        _repo_root, pipeline, cfg = _pipeline_context(args, require_gpu=True)
        runtime = _PrepareRuntime(pipeline=pipeline, cfg=cfg)
        try:
            _run_stage_loop(
                args=args,
                work_root=work_root,
                requests=requests,
                process_one=lambda request: _prepare_one(runtime, request, _stage_dir(work_root, "prepare", request)),
            )
        finally:
            runtime.close()
        return

    if args.stage == "merge":
        repo_root, _pipeline, cfg = _pipeline_context(args, require_gpu=False)
        pipeline = _init_merge_pipeline(repo_root, cfg)
        _run_stage_loop(
            args=args,
            work_root=work_root,
            requests=requests,
            process_one=lambda request: _merge_one(pipeline, work_root, request, _stage_dir(work_root, "merge", request)),
        )
        return

    if args.stage == "final":
        _repo_root, pipeline, cfg = _pipeline_context(args, require_gpu=True)
        runtime = _FinalRuntime(pipeline=pipeline, cfg=cfg)

        def process_final(request: dict[str, Any]) -> None:
            _final_one(runtime, work_root, request, args)
            if args.delete_after_final:
                _delete_intermediates(work_root, request)

        _run_stage_loop(
            args=args,
            work_root=work_root,
            requests=requests,
            process_one=process_final,
        )
        return

    raise ValueError(f"unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
