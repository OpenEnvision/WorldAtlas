from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any
from uuid import uuid4

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    latest_mp4,
    load_batch_spec,
    load_json,
    begin_sample,
    print_status,
)


WORLD_SCORE_CAMERA_TOKENS = frozenset(
    {
        "fixed",
        "push_in",
        "pull_out",
        "move_left",
        "move_right",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "pedestal_up",
        "pedestal_down",
        "roll_cw",
        "roll_ccw",
        "orbit_left",
        "orbit_right",
    }
)
WORLD_SCORE_PRIOR_PREFIX = "worldatlas-camera-path:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Stable Virtual Camera batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: Any, *, cast) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) > 1:
            return [cast(part) for part in parts]
        return cast(parts[0]) if parts else None
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return cast(value)


def _shared_checkpoint_root(checkpoint_dir: Path | None, generation: dict[str, Any]) -> Path | None:
    raw_root = generation.get("checkpoint_root")
    if raw_root:
        return Path(str(raw_root)).expanduser().resolve()
    if checkpoint_dir is not None:
        return checkpoint_dir.parent.resolve()
    return None


def _configure_runtime_env(
    *,
    checkpoint_dir: Path | None,
    generation: dict[str, Any],
) -> Path | None:
    ckpt_root = _shared_checkpoint_root(checkpoint_dir, generation)
    if ckpt_root is not None:
        hf_home = ckpt_root / "huggingface"
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(ckpt_root / "transformers")
        os.environ["TORCH_HOME"] = str(ckpt_root / "torch")
        for path in (hf_home, hf_home / "hub", ckpt_root / "transformers", ckpt_root / "torch"):
            path.mkdir(parents=True, exist_ok=True)

    if "CUDA_VISIBLE_DEVICES" not in os.environ and generation.get("gpu_index") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(generation["gpu_index"])
    if generation.get("offline") is not None:
        offline_value = "1" if _bool(generation.get("offline")) else "0"
        os.environ["HF_HUB_OFFLINE"] = offline_value
        os.environ["TRANSFORMERS_OFFLINE"] = offline_value
    return ckpt_root


def _patch_local_dependencies(ckpt_root: Path | None) -> None:
    if ckpt_root is None:
        return

    sd21_root = ckpt_root / "stable-diffusion-2-1-base"
    if sd21_root.is_dir():
        from diffusers.models import AutoencoderKL

        original_from_pretrained = AutoencoderKL.from_pretrained

        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            if str(pretrained_model_name_or_path) == "stabilityai/stable-diffusion-2-1-base":
                pretrained_model_name_or_path = str(sd21_root)
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        AutoencoderKL.from_pretrained = classmethod(from_pretrained)

    clip_root = ckpt_root / "CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_candidates = [
        clip_root / "open_clip_pytorch_model.bin",
        clip_root / "open_clip_model.safetensors",
    ]
    clip_weight = next((path for path in clip_candidates if path.is_file()), None)
    if clip_weight is not None:
        import open_clip

        original_create = open_clip.create_model_and_transforms

        def create_model_and_transforms(model_name, pretrained=None, *args, **kwargs):
            if str(model_name) == "ViT-H-14" and str(pretrained) == "laion2b_s32b_b79k":
                pretrained = str(clip_weight)
            return original_create(model_name, pretrained=pretrained, *args, **kwargs)

        open_clip.create_model_and_transforms = create_model_and_transforms


def _camera_path(row: dict[str, Any]) -> list[str]:
    camera_path = row.get("camera_path") or []
    if isinstance(camera_path, str):
        camera_path = [camera_path]
    normalized_path: list[str] = []
    for token in camera_path:
        normalized = str(token).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized:
            normalized_path.append(normalized)
    return normalized_path or ["fixed"]


def _camera_token(row: dict[str, Any]) -> str:
    """Return the primary token for status output and legacy override lookup."""
    return _camera_path(row)[0]


def _traj_prior(row: dict[str, Any], generation: dict[str, Any]) -> str:
    camera_path = _camera_path(row)
    unknown = [token for token in camera_path if token not in WORLD_SCORE_CAMERA_TOKENS]
    if unknown:
        raise ValueError(f"unsupported Stable Virtual Camera token(s): {unknown!r}")

    token = camera_path[0]
    overrides = generation.get("trajectory_by_camera_token")
    if len(camera_path) == 1 and isinstance(overrides, dict) and token in overrides:
        return str(overrides[token])
    configured = generation.get("traj_prior")
    if configured:
        return str(configured)
    return WORLD_SCORE_PRIOR_PREFIX + ",".join(camera_path)


def _patch_world_score_camera_prior(
    demo_module: Any,
    *,
    generation: dict[str, Any] | None = None,
) -> None:
    """Make SVC consume the exact camera convention used by WorldAtlas scoring.

    Upstream SVC only offers coarse presets: both orbit directions collapse to
    ``orbit``, and it has no in-place pan/tilt presets.  Reusing WorldAtlas's
    synthetic scoring trajectory keeps generation and evaluation semantics in
    lockstep and also supports the benchmark's full multi-segment camera path.
    """
    original_get_preset_pose_fov = demo_module.get_preset_pose_fov
    trajectory_runtime = {
        key: value
        for key, value in (generation or {}).items()
        if key.startswith("synthetic_camera_")
    }

    def get_preset_pose_fov(option, num_frames, start_w2c, look_at, *args, **kwargs):
        option_text = str(option)
        if not option_text.startswith(WORLD_SCORE_PRIOR_PREFIX):
            return original_get_preset_pose_fov(
                option,
                num_frames,
                start_w2c,
                look_at,
                *args,
                **kwargs,
            )

        camera_path = [
            token
            for token in option_text.removeprefix(WORLD_SCORE_PRIOR_PREFIX).split(",")
            if token
        ]
        unknown = [token for token in camera_path if token not in WORLD_SCORE_CAMERA_TOKENS]
        if not camera_path or unknown:
            raise ValueError(f"invalid WorldAtlas camera path prior: {option_text!r}")

        from worldatlas.benchmark.synthetic_camera import synthetic_camera_matrices

        relative_c2ws = synthetic_camera_matrices(
            camera_path,
            target_frames=int(num_frames),
            runtime=trajectory_runtime,
        ).matrices
        start_c2w = demo_module.torch.linalg.inv(start_w2c).detach().cpu().numpy()
        poses = start_c2w[None] @ relative_c2ws

        # Ask upstream for its default field of view while forcing it to remain
        # constant.  This preserves SVC's intrinsics contract without using its
        # incorrect zoom-out fallback for a fixed camera.
        fixed_kwargs = dict(kwargs)
        fixed_kwargs["zoom_factor"] = 1.0
        _, fovs = original_get_preset_pose_fov(
            "zoom-in",
            num_frames,
            start_w2c,
            look_at,
            *args,
            **fixed_kwargs,
        )
        return poses, fovs

    demo_module.get_preset_pose_fov = get_preset_pose_fov


def _prepare_input(row: dict[str, Any], data_dir: Path) -> Path:
    source = Path(str(row["conditioning_image"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"conditioning image not found: {source}")
    suffix = source.suffix if source.suffix else ".png"
    destination = data_dir / f"input{suffix}"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _demo_options(row: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    cfg = _as_list(generation.get("cfg", [4.0, 2.0]), cast=float)
    guider_types = _as_list(generation.get("guider_types", [1, 2]), cast=int)
    options = {
        "traj_prior": _traj_prior(row, generation),
        "chunk_strategy": str(generation.get("chunk_strategy", "interp")),
        "chunk_strategy_first_pass": str(generation.get("chunk_strategy_first_pass", "gt-nearest")),
        "num_targets": int(generation.get("num_targets", 20)),
        "video_save_fps": float(generation.get("video_save_fps", generation.get("fps", 12))),
        "num_steps": int(generation.get("num_steps", 50)),
        "cfg": cfg,
        "guider_types": guider_types,
        "cfg_min": float(generation.get("cfg_min", 1.2)),
        "camera_scale": float(generation.get("camera_scale", 2.0)),
        "replace_or_include_input": _bool(generation.get("replace_or_include_input"), True),
        "save_first_pass": _bool(generation.get("save_first_pass"), False),
        "save_second_pass": _bool(generation.get("save_second_pass"), False),
    }
    for key in (
        "transform_target",
        "transform_scale",
        "num_prior_frames",
        "num_prior_frames_ratio",
        "t_padding_mode",
        "ltr_first_pass",
        "zoom_factor",
    ):
        if key in generation:
            options[key] = generation[key]
    return options


def _load_model_once(
    *,
    demo_module: Any,
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> Any:
    """Load the heavy SGM diffusion model exactly once per shard process.

    The module-level ``AE`` / ``CONDITIONER`` / ``DENOISER`` are already created on
    import of ``demo``. Only the 5GB ``MODEL`` weight (``SGMWrapper(load_model(...))``)
    is re-created per call inside ``demo.main`` upstream, so we replicate just that
    load here and reuse the returned model for every row in this shard.
    """
    print(
        "[svc-batch-runner] loading SGM model once for this shard "
        f"(weight={generation.get('weight_name', 'modelv1.1.safetensors')}, "
        f"version={generation.get('model_version', 1.1)})",
        flush=True,
    )
    model = demo_module.SGMWrapper(
        demo_module.load_model(
            model_version=float(generation.get("model_version", 1.1)),
            pretrained_model_name_or_path=str(checkpoint_dir),
            weight_name=str(generation.get("weight_name", "modelv1.1.safetensors")),
            device="cpu",
            verbose=True,
        ).eval()
    ).to(demo_module.device)
    if getattr(demo_module, "COMPILE", False):
        model = demo_module.torch.compile(model, dynamic=False)
    print("[svc-batch-runner] SGM model loaded; will be reused for all rows", flush=True)
    return model


def _build_version_dict(row: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    """Build a fresh per-row VERSION_DICT mirroring ``demo.main``.

    ``parse_task``/``infer_prior_stats``/``run_one_scene`` mutate ``T``/``H``/``W`` in
    place, so each row needs its own dict.
    """
    version_dict = {
        "H": int(generation.get("H", generation.get("height", 576))),
        "W": int(generation.get("W", generation.get("width", 576))),
        "T": _as_list(generation.get("T", 21), cast=int) or 21,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": "nearest-gt",
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": 2.0,
            "camera_scale": 2.0,
            "num_steps": 50,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
        },
    }
    version_dict["options"].update(_demo_options(row, generation))
    return version_dict


def _run_scene(
    *,
    demo_module: Any,
    model: Any,
    scene: str,
    task: str,
    save_path_scene: str,
    version_dict: dict[str, Any],
    use_traj_prior: bool,
    seed: int,
) -> None:
    """Replicate the per-scene body of ``demo.main`` using a preloaded model."""
    np = demo_module.np
    torch = demo_module.torch
    glob = demo_module.glob
    osp = demo_module.osp

    options = version_dict["options"]
    num_inputs = options.get("num_inputs", None)

    (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        c2ws,
        Ks,
        anchor_c2ws,
        anchor_Ks,
    ) = demo_module.parse_task(
        task,
        scene,
        num_inputs,
        version_dict["T"],
        version_dict,
    )
    assert num_inputs is not None
    image_cond = {
        "img": all_imgs_path,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }
    camera_cond = {
        "c2w": c2ws.clone(),
        "K": Ks.clone(),
        "input_indices": list(range(num_inputs + num_targets)),
    }
    video_path_generator = demo_module.run_one_scene(
        task,
        version_dict,
        model=model,
        ae=demo_module.AE,
        conditioner=demo_module.CONDITIONER,
        denoiser=demo_module.DENOISER,
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=save_path_scene,
        use_traj_prior=use_traj_prior,
        traj_prior_Ks=anchor_Ks,
        traj_prior_c2ws=anchor_c2ws,
        seed=seed,
    )
    for _ in video_path_generator:
        pass

    c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()
    img_paths = sorted(glob.glob(osp.join(save_path_scene, "samples-rgb", "*.png")))
    if len(img_paths) != len(c2ws):
        input_img_paths = sorted(glob.glob(osp.join(save_path_scene, "input", "*.png")))
        assert len(img_paths) == num_targets
        assert len(input_img_paths) == num_inputs
        assert c2ws.shape[0] == num_inputs + num_targets
        target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
        img_paths = [
            input_img_paths[input_indices.index(i)]
            if i in input_indices
            else img_paths[target_indices.index(i)]
            for i in range(c2ws.shape[0])
        ]
    demo_module.create_transforms_simple(
        save_path=save_path_scene,
        img_paths=img_paths,
        img_whs=np.array([version_dict["W"], version_dict["H"]])[None].repeat(
            num_inputs + num_targets, 0
        ),
        c2ws=c2ws,
        Ks=Ks,
    )


def _generate_one(
    *,
    demo_module: Any,
    model: Any,
    row: dict[str, Any],
    row_index: int,
    generation: dict[str, Any],
    work_root: Path,
) -> Path:
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    sample_work = work_root / str(row.get("prediction_stem", output_path.stem))
    data_dir = sample_work / "data"
    scene = _prepare_input(row, data_dir)

    save_subdir = f"{output_path.stem}-{uuid4().hex[:8]}"
    task = str(generation.get("task", "img2trajvid_s-prob"))
    work_dir = str(work_root / "demo")
    save_path_scene = os.path.join(
        work_dir, task, save_subdir, os.path.splitext(os.path.basename(str(scene)))[0]
    )

    version_dict = _build_version_dict(row, generation)
    _run_scene(
        demo_module=demo_module,
        model=model,
        scene=str(scene),
        task=task,
        save_path_scene=save_path_scene,
        version_dict=version_dict,
        use_traj_prior=_bool(generation.get("use_traj_prior"), True),
        seed=int(generation.get("seed", 23)) + row_index,
    )

    scene_dir = Path(save_path_scene)
    generated = scene_dir / "samples-rgb.mp4"
    if not generated.is_file():
        generated = latest_mp4(scene_dir)
    copy_output(generated, output_path)

    # Keep per-shard disk usage bounded over hundreds of rows.
    for stale in (Path(work_dir) / task / save_subdir, sample_work):
        shutil.rmtree(stale, ignore_errors=True)
    return output_path


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Stable Virtual Camera checkpoint_dir not found: {checkpoint_dir}")

    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    ckpt_root = _configure_runtime_env(checkpoint_dir=checkpoint_dir, generation=generation)
    _patch_local_dependencies(ckpt_root)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import demo as svc_demo

    _patch_world_score_camera_prior(svc_demo, generation=generation)

    output_parents = {Path(str(row["output_path"])).expanduser().resolve().parent for row in rows}
    work_root = (next(iter(output_parents)) / "_stable_virtual_camera_work").resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    # Load the heavy diffusion model exactly ONCE for the whole shard and reuse it.
    model = _load_model_once(
        demo_module=svc_demo,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
    )

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        begin_sample(sample_id, index=row_index + 1, total=len(rows))
        requested_output = Path(str(row["output_path"])).expanduser().resolve()
        if requested_output.is_file() and requested_output.stat().st_size > 0:
            print_status(
                sample_id,
                "skipped_existing",
                output_path=str(requested_output),
                reason="output appeared after batch planning",
            )
            continue
        try:
            output_path = _generate_one(
                demo_module=svc_demo,
                model=model,
                row=row,
                row_index=row_index,
                generation=generation,
                work_root=work_root,
            )
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                camera_token=_camera_token(row),
                camera_path=_camera_path(row),
                traj_prior=_traj_prior(row, generation),
                control_source="worldatlas_score_camera_path",
            )
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
        finally:
            gc.collect()
    return 0 if failed < len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
