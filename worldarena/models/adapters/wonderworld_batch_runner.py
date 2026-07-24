"""Batch subprocess runner for WonderWorld inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import traceback
import time
from typing import Any
import warnings
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from worldarena.common.checkpoints import apply_checkpoint_env


XYZ_SCALE = 1000
BACKGROUND_RGB = (0.7, 0.7, 0.7)
BACKGROUND_TOLERANCE = 12.0 / 255.0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena WonderWorld batch runner.")
    parser.add_argument("--batch_spec", default=None, type=str)
    parser.add_argument("--batch_results", default=None, type=str)
    parser.add_argument("--repo_root", default=None, type=str)
    parser.add_argument("--entrypoint", default="run.py", type=str)
    parser.add_argument("--conditioning_image", default=None, type=str)
    parser.add_argument("--annotation_path", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--archive_path", default=None, type=str)
    parser.add_argument("--sample_name", default=None, type=str)
    parser.add_argument("--prompt_sequence_json", default=None, type=str)
    parser.add_argument("--style_prompt", default="photorealistic", type=str)
    parser.add_argument("--repvit_checkpoint", default=None, type=str)
    parser.add_argument("--stable_diffusion_checkpoint", default="sd2-community/stable-diffusion-2-inpainting")
    parser.add_argument("--depth_model_repo", default="prs-eth/marigold-v1-0", type=str)
    parser.add_argument("--normal_model_repo", default="prs-eth/marigold-normals-v0-1", type=str)
    parser.add_argument("--oneformer_model_repo", default="shi-labs/oneformer_ade20k_swin_large", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_scenes", default=1, type=int)
    parser.add_argument("--trajectory_frames", default=49, type=int)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--depth_model", default="marigold", type=str)
    parser.add_argument("--camera_speed", default=0.001, type=float)
    parser.add_argument("--fg_depth_range", default=0.015, type=float)
    parser.add_argument("--depth_shift", default=0.001, type=float)
    parser.add_argument("--sky_hard_depth", default=0.02, type=float)
    parser.add_argument("--init_focal_length", default=960.0, type=float)
    parser.add_argument("--inpainting_resolution_gen", default=512, type=int)
    parser.add_argument("--inpainting_resolution_interp", default=512, type=int)
    parser.add_argument("--rotation_path", default=None, type=str)
    parser.add_argument("--min_render_content_fraction", default=0.02, type=float)
    parser.add_argument("--max_render_blank_frame_ratio", default=0.2, type=float)
    parser.add_argument("--use_gpt", default=False, type=_parse_bool)
    parser.add_argument("--debug", default=False, type=_parse_bool)
    parser.add_argument("--gen_layer", default=False, type=_parse_bool)
    parser.add_argument("--use_compile", default=False, type=_parse_bool)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    return parser.parse_args(argv)


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_wonderworld_batch_"))
        try:
            yield path
        finally:
            pass
        return
    with tempfile.TemporaryDirectory(prefix="worldarena_wonderworld_batch_") as temp_dir:
        yield Path(temp_dir)


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def _normalized_env(repo_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = apply_checkpoint_env(base_env or os.environ.copy())
    pythonpaths = [
        str(repo_root),
        str(repo_root / "RepViT"),
        str(repo_root / "RepViT" / "sam"),
    ]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _require_repo_layout(repo_root: Path, entrypoint: str) -> tuple[Path, Path]:
    entrypoint_path = (repo_root / entrypoint).resolve()
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"WonderWorld entrypoint not found: {entrypoint_path}")
    base_config_path = (repo_root / "config" / "base-config.yaml").resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"WonderWorld base config not found: {base_config_path}")
    return entrypoint_path, base_config_path


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _parse_rotation_path(value: str | None, *, num_scenes: int) -> list[int]:
    if value is None or not str(value).strip():
        return [2] * max(int(num_scenes), 1)
    text = str(value).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise ValueError(f"rotation_path must be a list or comma-separated string: {value!r}")
    values = [int(item) for item in parsed]
    if len(values) < num_scenes:
        values.extend([values[-1] if values else 2] * (num_scenes - len(values)))
    return values[:num_scenes]


def _prompt_sequence(raw: str, *, num_scenes: int) -> list[str]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("prompt_sequence_json must decode to a list")
    prompts = [" ".join(str(item).split()) for item in payload if str(item).strip()]
    if not prompts:
        prompts = ["photorealistic scene"]
    while len(prompts) < num_scenes + 1:
        prompts.append(prompts[-1])
    return prompts[: num_scenes + 1]


def _write_example_config(path: Path, *, args: argparse.Namespace, example_name: str) -> Path:
    payload = {
        "runs_dir": f"output/{example_name}",
        "example_name": example_name,
        "seed": int(args.seed),
        "num_scenes": int(args.num_scenes),
        "use_gpt": bool(args.use_gpt),
        "debug": bool(args.debug),
        "depth_model": str(args.depth_model),
        "camera_speed": float(args.camera_speed),
        "fg_depth_range": float(args.fg_depth_range),
        "depth_shift": float(args.depth_shift),
        "sky_hard_depth": float(args.sky_hard_depth),
        "init_focal_length": float(args.init_focal_length),
        "inpainting_resolution_gen": int(args.inpainting_resolution_gen),
        "inpainting_resolution_interp": int(args.inpainting_resolution_interp),
        "stable_diffusion_checkpoint": str(args.stable_diffusion_checkpoint),
        "rotation_path": _parse_rotation_path(args.rotation_path, num_scenes=int(args.num_scenes)),
        "gen_layer": bool(args.gen_layer),
        "use_compile": bool(args.use_compile),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _load_pose_sequence(annotation_path: Path) -> np.ndarray:
    poses_path = annotation_path / "poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"WonderWorld batch mode requires poses.npy: {poses_path}")
    poses = np.load(poses_path).astype(np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected poses.npy with shape [N,4,4], got {poses.shape}")
    return poses


def _matrix_to_pose_vector(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    quat /= np.linalg.norm(quat) + 1e-8
    return np.concatenate([matrix[:3, 3].astype(np.float32), quat]).astype(np.float32)


def _quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = (quat / (np.linalg.norm(quat) + 1e-8)).astype(np.float64)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _pose_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quaternion_to_rotation_matrix(vector[3:7])
    matrix[:3, 3] = vector[:3]
    return matrix


def _slerp(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    start = start / (np.linalg.norm(start) + 1e-8)
    end = end / (np.linalg.norm(end) + 1e-8)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = start + ratio * (end - start)
        return (blended / (np.linalg.norm(blended) + 1e-8)).astype(np.float32)
    theta_0 = np.arccos(dot)
    theta = theta_0 * ratio
    return (
        (np.sin(theta_0 - theta) / np.sin(theta_0)) * start
        + (np.sin(theta) / np.sin(theta_0)) * end
    ).astype(np.float32)


def _interpolate_poses(key_poses: np.ndarray, frame_count: int) -> np.ndarray:
    if len(key_poses) <= 1:
        return np.repeat(key_poses[:1], max(int(frame_count), 1), axis=0).astype(np.float32)
    vectors = np.stack([_matrix_to_pose_vector(pose) for pose in key_poses], axis=0)
    target_frames = max(int(frame_count), len(key_poses))
    key_positions = np.linspace(0.0, 1.0, num=len(vectors), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    rows: list[np.ndarray] = []
    for position in target_positions:
        idx = int(np.searchsorted(key_positions, position, side="right") - 1)
        idx = min(max(idx, 0), len(vectors) - 2)
        local = (position - key_positions[idx]) / (key_positions[idx + 1] - key_positions[idx] + 1e-8)
        translation = (1.0 - local) * vectors[idx, :3] + local * vectors[idx + 1, :3]
        rotation = _slerp(vectors[idx, 3:7], vectors[idx + 1, 3:7], float(local))
        rows.append(np.concatenate([translation, rotation]).astype(np.float32))
    return np.stack([_pose_vector_to_matrix(row) for row in rows], axis=0).astype(np.float32)


def _camera_from_pose(pose: np.ndarray, *, focal_length: float, device):
    from pytorch3d.renderer import PerspectiveCameras

    c2w = np.asarray(pose, dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    rotation = torch.tensor(w2c[:3, :3].T, dtype=torch.float32, device=device).unsqueeze(0)
    translation = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=device).unsqueeze(0)
    k = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    k[0, 0, 0] = float(focal_length)
    k[0, 1, 1] = float(focal_length)
    k[0, 0, 2] = 256.0
    k[0, 1, 2] = 256.0
    k[0, 2, 3] = 1.0
    k[0, 3, 2] = 1.0
    return PerspectiveCameras(
        K=k,
        R=rotation,
        T=translation,
        in_ndc=False,
        image_size=((512, 512),),
        device=device,
    )


def _load_cameras(annotation_path: Path, *, num_scenes: int, trajectory_frames: int, focal_length: float):
    poses = _load_pose_sequence(annotation_path)
    key_count = max(int(num_scenes) + 1, 2)
    if len(poses) < key_count:
        poses = _interpolate_poses(poses, key_count)
    key_indices = np.linspace(0, len(poses) - 1, num=key_count).round().astype(int)
    key_poses = poses[key_indices]
    interp_poses = _interpolate_poses(key_poses, trajectory_frames)
    device = torch.device("cuda")
    return (
        [_camera_from_pose(pose, focal_length=focal_length, device=device) for pose in key_poses],
        [_camera_from_pose(pose, focal_length=focal_length, device=device) for pose in interp_poses],
    )


def _seed_everything(seed: int) -> None:
    if seed == -1:
        seed = int(np.random.randint(2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def _write_video(output_path: Path, frames: list[Image.Image], fps: float) -> None:
    if not frames:
        raise RuntimeError(f"WonderWorld produced no rendered frames for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0].convert("RGB"), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(first.shape[1]), int(first.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open WonderWorld video writer: {output_path}")
    for frame in frames:
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def _validate_rendered_frames(
    frames: list[Image.Image],
    *,
    min_content_fraction: float,
    max_blank_frame_ratio: float,
) -> dict[str, object]:
    """Reject renders dominated by WonderWorld's constant empty background."""
    if not frames:
        raise RuntimeError("WonderWorld produced no rendered frames")
    min_content_fraction = float(min_content_fraction)
    max_blank_frame_ratio = float(max_blank_frame_ratio)
    if not 0.0 <= min_content_fraction <= 1.0:
        raise ValueError(
            "min_render_content_fraction must be in [0, 1], "
            f"got {min_content_fraction!r}"
        )
    if not 0.0 <= max_blank_frame_ratio <= 1.0:
        raise ValueError(
            "max_render_blank_frame_ratio must be in [0, 1], "
            f"got {max_blank_frame_ratio!r}"
        )

    background = np.asarray(BACKGROUND_RGB, dtype=np.float32).reshape(1, 1, 3)
    content_fractions: list[float] = []
    for frame in frames:
        rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        differs_from_background = np.max(np.abs(rgb - background), axis=2) > BACKGROUND_TOLERANCE
        content_fractions.append(float(np.mean(differs_from_background)))

    blank_frame_count = sum(value < min_content_fraction for value in content_fractions)
    blank_frame_ratio = blank_frame_count / len(content_fractions)
    summary: dict[str, object] = {
        "frame_count": len(content_fractions),
        "blank_frame_count": blank_frame_count,
        "blank_frame_ratio": round(blank_frame_ratio, 6),
        "min_content_fraction": round(min(content_fractions), 6),
        "median_content_fraction": round(float(np.median(content_fractions)), 6),
        "max_content_fraction": round(max(content_fractions), 6),
        "content_fraction_threshold": min_content_fraction,
        "blank_frame_ratio_threshold": max_blank_frame_ratio,
    }
    if blank_frame_ratio > max_blank_frame_ratio:
        raise RuntimeError(
            "WonderWorld render validation failed: "
            f"{blank_frame_count}/{len(content_fractions)} frames contain less than "
            f"{min_content_fraction:.2%} non-background content "
            f"(blank ratio {blank_frame_ratio:.2%}, allowed {max_blank_frame_ratio:.2%})"
        )
    return summary


def _write_frames(frames_dir: Path, frames: list[Image.Image]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.convert("RGB").save(frames_dir / f"{index:05d}.png")


def _archive_sidecar(path: Path, *, work_dir: Path, frames_dir: Path, example_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_path in sorted(frames_dir.glob("*.png")):
            archive.write(frame_path, f"frames/{frame_path.name}")
        for source in sorted((work_dir / "output" / example_name).glob("**/*")):
            if source.is_file() and source.suffix.lower() in {".yaml", ".txt", ".png", ".ply", ".splat", ".pth"}:
                archive.write(source, f"run/{source.relative_to(work_dir / 'output' / example_name)}")


def _run_batch_generation(args: argparse.Namespace, *, work_dir: Path) -> list[Image.Image]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    pythonpath_roots = [str(repo_root), str(repo_root / "RepViT"), str(repo_root / "RepViT" / "sam")]
    sys.path[:] = pythonpath_roots + [path for path in sys.path if path not in pythonpath_roots]
    os.chdir(work_dir)
    warnings.filterwarnings("ignore")

    from diffusers import DDIMScheduler, EulerDiscreteScheduler
    from diffusers.models.attention_processor import AttnProcessor2_0
    from gaussian_renderer import render
    from kornia.morphology import dilation
    from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline, MarigoldPipeline

    import syncdiffusion.syncdiffusion_model as syncdiffusion_model
    from omegaconf import OmegaConf
    from scene import GaussianModel, Scene
    from torchvision.transforms import ToPILImage, ToTensor
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
    from util.segment_utils import create_mask_generator_repvit
    from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from util.utils import convert_pt3d_cam_to_3dgs_cam, prepare_scheduler, save_depth_map, soft_stitching
    from arguments import GSParams
    from models.models import KeyframeGen
    from utils.loss import l1_loss, ssim

    del save_depth_map

    base_config = OmegaConf.load(str((repo_root / "config" / "base-config.yaml").resolve()))
    example_config = OmegaConf.load(str(work_dir / "config" / "worldarena_example.yaml"))
    config = OmegaConf.merge(base_config, example_config)
    config.num_scenes = int(args.num_scenes)

    _seed_everything(int(config["seed"]))
    background = torch.tensor(BACKGROUND_RGB, dtype=torch.float32, device="cuda")
    start_keyframe = Image.open(args.conditioning_image).convert("RGB").resize((512, 512))
    prompt_list = _prompt_sequence(args.prompt_sequence_json, num_scenes=int(args.num_scenes))
    cameras, cameras_interp = _load_cameras(
        Path(args.annotation_path).expanduser().resolve(),
        num_scenes=int(args.num_scenes),
        trajectory_frames=int(args.trajectory_frames),
        focal_length=float(args.init_focal_length),
    )

    segment_processor = OneFormerProcessor.from_pretrained(str(args.oneformer_model_repo))
    segment_model = OneFormerForUniversalSegmentation.from_pretrained(
        str(args.oneformer_model_repo)
    ).to("cuda")
    mask_generator = create_mask_generator_repvit()
    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        config["stable_diffusion_checkpoint"],
        safety_checker=None,
        torch_dtype=torch.bfloat16,
    ).to(config["device"])
    inpainter_pipeline.scheduler = DDIMScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())
    if bool(config.get("use_compile", False)):
        torch._inductor.config.conv_1x1_as_mm = True
        inpainter_pipeline.unet.to(memory_format=torch.channels_last)
        inpainter_pipeline.vae.to(memory_format=torch.channels_last)
        inpainter_pipeline.unet = torch.compile(inpainter_pipeline.unet)
        inpainter_pipeline.vae.decode = torch.compile(inpainter_pipeline.vae.decode)

    depth_model = MarigoldPipeline.from_pretrained(
        str(args.depth_model_repo),
        torch_dtype=torch.bfloat16,
    ).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)
    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        str(args.normal_model_repo),
        torch_dtype=torch.bfloat16,
    ).to(config["device"])

    kf_gen = KeyframeGen(
        config=config,
        inpainter_pipeline=inpainter_pipeline,
        mask_generator=mask_generator,
        depth_model=depth_model,
        segment_model=segment_model,
        segment_processor=segment_processor,
        normal_estimator=normal_estimator,
        rotation_path=list(config["rotation_path"])[: int(config["num_scenes"])],
        inpainting_resolution=config["inpainting_resolution_gen"],
    ).to(config["device"])
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config["device"])
    sky_mask = kf_gen.generate_sky_mask().float()
    sky_dir = Path("examples") / "sky_images" / str(config["example_name"])
    needs_sky_image = bool(config.get("gen_sky_image", False))
    if needs_sky_image or not (sky_dir / "sky_0.png").exists() or not (sky_dir / "sky_1.png").exists():
        from syncdiffusion.syncdiffusion_model import SyncDiffusion

        syncdiffusion_model = SyncDiffusion(config["device"], sd_version="2.0-inpaint", hf_key=config["stable_diffusion_checkpoint"])
    else:
        syncdiffusion_model = None
    kf_gen.generate_sky_pointcloud(
        syncdiffusion_model,
        image=kf_gen.image_latest,
        mask=sky_mask,
        gen_sky=needs_sky_image,
        style=str(args.style_prompt or "photorealistic"),
    )
    kf_gen.recompose_image_latest_and_set_current_pc(scene_name=prompt_list[0])
    pt_gen = None
    if bool(config.get("use_gpt", False)):
        from util.chatGPT4 import TextpromptGen

        pt_gen = TextpromptGen(kf_gen.run_dir, False)
    kf_gen.increment_kf_idx()

    traindatas = kf_gen.convert_to_3dgs_traindata(
        xyz_scale=XYZ_SCALE,
        remove_threshold=None,
        use_no_loss_mask=False,
    )
    if bool(config["gen_layer"]):
        _, traindata_sky, _ = traindatas
    else:
        _, traindata_sky = traindatas

    sky_opt = GSParams()
    sky_opt.max_screen_size = 100
    sky_opt.scene_extent = 1.5
    sky_opt.densify_from_iter = 200
    sky_opt.prune_from_iter = 200
    sky_opt.densify_grad_threshold = 1.0
    sky_opt.iterations = 399
    gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
    scene = Scene(traindata_sky, gaussians, sky_opt, is_sky=True)
    _train_gaussian(
        gaussians,
        scene,
        sky_opt,
        background,
        render,
        l1_loss,
        ssim,
        initialize_scaling=False,
    )

    gaussians.visibility_filter_all = torch.zeros(
        gaussians.get_xyz_all.shape[0],
        dtype=torch.bool,
        device="cuda",
    )
    gaussians.delete_mask_all = torch.zeros(
        gaussians.get_xyz_all.shape[0],
        dtype=torch.bool,
        device="cuda",
    )
    gaussians.is_sky_filter = torch.ones(
        gaussians.get_xyz_all.shape[0],
        dtype=torch.bool,
        device="cuda",
    )

    opt = GSParams()
    if bool(config["gen_layer"]):
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(
            xyz_scale=XYZ_SCALE
        )
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata_layer, gaussians, opt)
        _train_gaussian(gaussians, scene, opt, background, render, l1_loss, ssim)
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=XYZ_SCALE,
            use_no_loss_mask=False,
        )

    if traindata["pcd_points"].shape[-1] != 0:
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        _train_gaussian(gaussians, scene, opt, background, render, l1_loss, ssim)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=XYZ_SCALE)
    gaussians.set_inscreen_points_to_visible(tdgs_cam)
    adaptive_negative_prompt = ""

    for scene_index in range(1, int(config["num_scenes"]) + 1):
        inpainting_prompt = prompt_list[scene_index]
        if bool(config["use_gpt"]):
            del pt_gen
        kf_gen.set_kf_param(
            inpainting_resolution=config["inpainting_resolution_gen"],
            inpainting_prompt=inpainting_prompt,
            adaptive_negative_prompt=adaptive_negative_prompt,
        )
        current_pt3d_cam = cameras[scene_index]
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=XYZ_SCALE)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)

        with torch.no_grad():
            render_pkg = render(tdgs_cam, gaussians, opt, background)
            render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)
        side_sky_height = 128
        sky_cond_width = 40
        inpaint_mask_0p5_nosky = render_pkg_nosky["final_opacity"] < 0.6
        inpaint_mask_0p0_nosky = render_pkg_nosky["final_opacity"] < 0.01
        inpaint_mask_0p5 = render_pkg["final_opacity"] < 0.6
        inpaint_mask_0p0 = render_pkg["final_opacity"] < 0.01
        fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
        foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1) > 150
        foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]
        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config["device"])
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[:, :, :, min_index : max_index + 1] = 1
        mask_using_full_render[:, :, :sky_cond_width, :] = 1
        mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
        mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
        mask_using_nosky_render = 1 - mask_using_full_render
        outpaint_condition_image = (
            render_pkg_nosky["render"] * mask_using_nosky_render
            + render_pkg["render"] * mask_using_full_render
        )
        fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
        outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
        outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
        kf_gen.inpaint(
            outpaint_condition_image,
            inpaint_mask=outpaint_mask,
            fill_mask=fill_mask,
            inpainting_prompt=inpainting_prompt,
            mask_strategy=np.max,
            diffusion_steps=50,
        )

        sem_seg = kf_gen.update_sky_mask()
        recomposed = soft_stitching(render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)
        depth_should_be = render_pkg["median_depth"][0:1].unsqueeze(0) / XYZ_SCALE
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)
        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (
            depth_should_be_ground < 0.006 * 0.8
        )
        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)
        with torch.no_grad():
            kf_gen.get_depth(
                kf_gen.image_latest,
                target_depth=depth_should_be_joint,
                mask_align=joint_mask,
                archive_output=True,
                diffusion_steps=30,
                guidance_steps=30,
            )
        kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
        kf_gen.image_latest = recomposed
        valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
        kf_gen.update_current_pc_by_kf(
            image=kf_gen.image_latest,
            depth=kf_gen.depth_latest,
            valid_mask=valid_px_mask,
        )
        kf_gen.archive_latest()

        traindata = kf_gen.convert_to_3dgs_traindata_latest(
            xyz_scale=XYZ_SCALE,
            use_no_loss_mask=False,
        )
        if traindata["pcd_points"].shape[-1] != 0:
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata, gaussians, opt)
            _train_gaussian(gaussians, scene, opt, background, render, l1_loss, ssim)
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
        kf_gen.increment_kf_idx()

    rendered_frames: list[Image.Image] = []
    for camera in cameras_interp:
        tdgs_cam_tmp = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=XYZ_SCALE)
        render_pkg_tmp = render(tdgs_cam_tmp, gaussians, opt, background)
        rendered_frames.append(ToPILImage()(render_pkg_tmp["render"]))
    return rendered_frames


def _train_gaussian(
    gaussians,
    scene,
    opt,
    background,
    render,
    l1_loss,
    ssim,
    *,
    initialize_scaling: bool = True,
) -> None:
    train_cameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=train_cameras, initialize_scaling=initialize_scaling)
    for iteration in range(1, opt.iterations + 1):
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        loss.backward()
        n_trainable = gaussians.get_xyz.shape[0]
        viewspace_grad = viewspace_point_tensor.grad[:n_trainable]
        visibility_filter = visibility_filter[:n_trainable]
        radii = radii[:n_trainable]
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_grad, visibility_filter)
                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    scene_extent = 0.0003 * XYZ_SCALE * 2 if opt.scene_extent is None else opt.scene_extent
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.05,
                        scene_extent,
                        max_screen_size,
                    )
                    gaussians.compute_3D_filter(cameras=train_cameras)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def _validate_single_args(args: argparse.Namespace) -> None:
    required = (
        "repo_root",
        "conditioning_image",
        "annotation_path",
        "output_path",
        "sample_name",
        "prompt_sequence_json",
        "repvit_checkpoint",
    )
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError("missing required WonderWorld runner arguments: " + ", ".join(missing))


def _cache_key(name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    payload = {
        "name": name,
        "args": [str(item) for item in args],
        "kwargs": {key: str(value) for key, value in sorted(kwargs.items())},
    }
    return json.dumps(payload, sort_keys=True)


def _install_singleload_caches() -> None:
    """Install singleload caches."""
    cache: dict[str, object] = {}
    for module_name in list(sys.modules):
        if module_name in {"utils", "util", "scene"} or module_name.startswith(("utils.", "util.", "scene.")):
            sys.modules.pop(module_name, None)

    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
    import util.segment_utils as segment_utils
    from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline, MarigoldPipeline

    import syncdiffusion.syncdiffusion_model as syncdiffusion_model

    original_processor_from_pretrained = OneFormerProcessor.from_pretrained

    @classmethod
    def cached_processor_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("oneformer_processor", args, kwargs)
        if key not in cache:
            print(f"[WonderWorld singleload] loading OneFormerProcessor once: {args[0] if args else ''}", flush=True)
            cache[key] = original_processor_from_pretrained(*args, **kwargs)
        return cache[key]

    OneFormerProcessor.from_pretrained = cached_processor_from_pretrained

    original_segment_from_pretrained = OneFormerForUniversalSegmentation.from_pretrained

    @classmethod
    def cached_segment_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("oneformer_model", args, kwargs)
        if key not in cache:
            print(f"[WonderWorld singleload] loading OneFormer model once: {args[0] if args else ''}", flush=True)
            cache[key] = original_segment_from_pretrained(*args, **kwargs)
        return cache[key]

    OneFormerForUniversalSegmentation.from_pretrained = cached_segment_from_pretrained

    original_mask_generator = segment_utils.create_mask_generator_repvit

    def cached_mask_generator(*args, **kwargs):
        key = _cache_key("repvit_mask_generator", args, kwargs)
        if key not in cache:
            print("[WonderWorld singleload] loading RepViT mask generator once", flush=True)
            cache[key] = original_mask_generator(*args, **kwargs)
        return cache[key]

    segment_utils.create_mask_generator_repvit = cached_mask_generator

    original_inpaint_from_pretrained = StableDiffusionInpaintPipeline.from_pretrained

    @classmethod
    def cached_inpaint_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("stable_diffusion_inpaint", args, kwargs)
        if key not in cache:
            print(f"[WonderWorld singleload] loading inpaint pipeline once: {args[0] if args else ''}", flush=True)
            cache[key] = original_inpaint_from_pretrained(*args, **kwargs)
        return cache[key]

    StableDiffusionInpaintPipeline.from_pretrained = cached_inpaint_from_pretrained

    original_depth_from_pretrained = MarigoldPipeline.from_pretrained

    @classmethod
    def cached_depth_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("marigold_depth", args, kwargs)
        if key not in cache:
            print(f"[WonderWorld singleload] loading Marigold depth once: {args[0] if args else ''}", flush=True)
            cache[key] = original_depth_from_pretrained(*args, **kwargs)
        return cache[key]

    MarigoldPipeline.from_pretrained = cached_depth_from_pretrained

    original_normal_from_pretrained = MarigoldNormalsPipeline.from_pretrained

    @classmethod
    def cached_normal_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("marigold_normals", args, kwargs)
        if key not in cache:
            print(f"[WonderWorld singleload] loading Marigold normals once: {args[0] if args else ''}", flush=True)
            cache[key] = original_normal_from_pretrained(*args, **kwargs)
        return cache[key]

    MarigoldNormalsPipeline.from_pretrained = cached_normal_from_pretrained


    original_syncdiffusion = syncdiffusion_model.SyncDiffusion

    def cached_syncdiffusion(*args, **kwargs):
        key = _cache_key("syncdiffusion", args, kwargs)
        if key not in cache:
            print("[WonderWorld singleload] loading SyncDiffusion once", flush=True)
            cache[key] = original_syncdiffusion(*args, **kwargs)
        return cache[key]

    syncdiffusion_model.SyncDiffusion = cached_syncdiffusion


def _run_one(args: argparse.Namespace) -> dict[str, object]:
    _validate_single_args(args)
    repo_root = Path(args.repo_root).expanduser().resolve()
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    repvit_checkpoint = Path(args.repvit_checkpoint).expanduser().resolve()
    _require_repo_layout(repo_root, args.entrypoint)
    if not conditioning_image.exists():
        raise FileNotFoundError(f"WonderWorld conditioning image not found: {conditioning_image}")
    if not repvit_checkpoint.exists():
        raise FileNotFoundError(f"WonderWorld RepViT checkpoint not found: {repvit_checkpoint}")

    started_at = time.perf_counter()
    example_name = _slugify(str(args.sample_name))
    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        local_conditioning = work_dir / "inputs" / f"conditioning{conditioning_image.suffix or '.png'}"
        local_conditioning.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conditioning_image, local_conditioning)
        _link_or_copy(repvit_checkpoint, work_dir / "repvit_sam.pt")
        _write_example_config(work_dir / "config" / "worldarena_example.yaml", args=args, example_name=example_name)

        old_cwd = Path.cwd()
        old_environ = os.environ.copy()
        runtime_env = _normalized_env(repo_root, old_environ)
        os.environ.clear()
        os.environ.update(runtime_env)
        print(f"[WonderWorld batch] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
        try:
            frames = _run_batch_generation(args, work_dir=work_dir)
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_environ)

        render_validation = _validate_rendered_frames(
            frames,
            min_content_fraction=float(args.min_render_content_fraction),
            max_blank_frame_ratio=float(args.max_render_blank_frame_ratio),
        )

        frames_dir = output_path.with_suffix("")
        frames_dir = frames_dir.with_name(f"{frames_dir.name}.frames")
        _write_frames(frames_dir, frames)
        _write_video(output_path, frames, fps=float(args.fps))
        if args.archive_path:
            _archive_sidecar(
                Path(args.archive_path).expanduser().resolve(),
                work_dir=work_dir,
                frames_dir=frames_dir,
                example_name=example_name,
            )
    result: dict[str, object] = {
        "status": "generated",
        "prediction_path": str(output_path),
        "generation_wall_time_seconds": round(time.perf_counter() - started_at, 6),
        "render_validation": render_validation,
    }
    if args.archive_path and Path(args.archive_path).expanduser().exists():
        result["archive_path"] = str(Path(args.archive_path).expanduser().resolve())
    return result


def _run_batch_spec(args: argparse.Namespace) -> None:
    spec_path = Path(str(args.batch_spec)).expanduser().resolve()
    results_path = (
        Path(str(args.batch_results)).expanduser().resolve()
        if args.batch_results
        else spec_path.with_suffix(".results.json")
    )
    rows = [json.loads(line) for line in spec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: dict[str, dict[str, object]] = {}
    if rows:
        first_args = parse_args(list(rows[0]["args"]))
        repo_root = Path(first_args.repo_root).expanduser().resolve()
        pythonpath_roots = [str(repo_root), str(repo_root / "RepViT"), str(repo_root / "RepViT" / "sam")]
        sys.path[:] = pythonpath_roots + [path for path in sys.path if path not in pythonpath_roots]
        _install_singleload_caches()

    for row in rows:
        sample_id = str(row["sample_id"])
        request_args = parse_args(list(row["args"]))
        print(f"[WonderWorld batch] generating {sample_id} -> {request_args.output_path}", flush=True)
        try:
            payload = _run_one(request_args)
            payload.setdefault("prompt", row.get("prompt", ""))
        except Exception as exc:
            payload = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "prediction_path": row.get("prediction_path"),
                "prompt": row.get("prompt", ""),
            }
        results[sample_id] = payload
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_spec:
        _run_batch_spec(args)
        return
    _run_one(args)


if __name__ == "__main__":
    main()
