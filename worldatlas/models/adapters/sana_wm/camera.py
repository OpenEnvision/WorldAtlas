"""Sana-WM controls using its unmodified official camera integrator."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np

from worldatlas.benchmark.synthetic_camera import _normalize_camera_path, segment_interval_counts
from worldatlas.common.checkpoints import project_root
from worldatlas.common.media import probe_image
from worldatlas.models.adapters.lingbot_world.common import _camera_path_for_sample
from worldatlas.models.adapters.pose_synthesis import _synthetic_intrinsics_sequence, local_pose_annotation_path

RECIPE = "sana_wm_official_demo_v1"
CONTROLS = {
    "fixed": set(), "push_in": {"forward"}, "pull_out": {"back"},
    "move_left": {"strafe_left"}, "move_right": {"strafe_right"},
    "pan_left": {"yaw_left"}, "pan_right": {"yaw_right"},
    "tilt_up": {"pitch_up"}, "tilt_down": {"pitch_down"},
    # Orbit is a benchmark extension composed from two supported native keys.
    "orbit_left": {"strafe_left", "yaw_right"},
    "orbit_right": {"strafe_right", "yaw_left"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def official_controller(repo_root: Path):
    path = Path(repo_root).resolve() / "inference_video_scripts/wm/camera_control.py"
    name = "_worldatlas_sana_camera_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def demo_camera_trajectory(camera_path, *, frame_count: int, suite: str, repo_root: Path):
    if frame_count < 9 or (frame_count - 1) % 8:
        raise ValueError("Sana camera controls require 8k+1 frames")
    if suite not in {"image_static", "image_dynamic", "memory_loop"}:
        raise ValueError(f"Unsupported Sana camera suite: {suite}")
    tokens = _normalize_camera_path(camera_path)
    if unknown := set(tokens) - CONTROLS.keys():
        raise ValueError(f"Unsupported Sana native camera actions: {sorted(unknown)}")
    if suite == "image_dynamic" and tokens != ["fixed"]:
        raise ValueError("Sana Image Dynamic camera must remain fixed")
    intervals = segment_interval_counts(len(tokens), frame_count)
    if min(intervals) < 1:
        raise ValueError("Not enough camera frames for the action sequence")
    memory = suite == "memory_loop"
    if memory and len(set(intervals)) != 1:
        raise ValueError("Memory requires equal action durations to preserve paired displacements")
    core = official_controller(repo_root)
    if (core.FPS, core.DEFAULT_TRANSLATION_SPEED, core.DEFAULT_ROTATION_SPEED_DEG) != (16, .025, .6):
        raise ValueError("Sana official camera defaults changed; review this versioned recipe")
    integrator = core.CameraPoseIntegrator()
    poses = [integrator.pose.copy()]
    angles = []
    for token, count in zip(tokens, intervals):
        degrees = (20. if token.startswith("tilt_") else 30.) if memory else .6 * count
        angles.append(degrees)
        velocity = core.controls_to_target_velocity(
            CONTROLS[token], translation_speed=.025, rotation_speed_rad=math.radians(degrees / count))
        # Held keys match official inference exactly. At explicit benchmark
        # segment boundaries, stop/switch immediately so fixed stays fixed and
        # inverse Memory actions close; do not coast into the following task.
        for _ in range(count):
            poses.append(integrator.step(velocity))
    matrices = np.asarray(poses, dtype=np.float32)
    metadata = dict(recipe=RECIPE, camera_path=tokens, frame_count=frame_count, suite=suite,
        fps=16, translation_per_frame=.025, image_rotation_degrees_per_frame=.6,
        segment_intervals=list(intervals), rotation_degrees_per_segment=angles,
        memory_preserves_task_turn_angles=memory, boundary_policy="immediate_task_action_switch",
        orbit_policy="native_strafe_plus_opposite_yaw", pitch_limit_degrees=core.DEFAULT_PITCH_LIMIT_DEG,
        coordinate_convention="OpenCV c2w; x right, y down, z forward; native scene units",
        official_controller_sha256=sha(Path(core.__file__)))
    return matrices, metadata


def _save_verified(path: Path, array: np.ndarray) -> None:
    if path.is_file():
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise ValueError(f"Changed cached Sana camera: {path}")
    else:
        temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.npy")
        np.save(temporary, array)
        temporary.replace(path)


def demo_camera_annotation(sample, conditioning_image: Path, *, frame_count: int,
                           focal_scale: float, repo_root: Path):
    annotation = local_pose_annotation_path(sample)
    if annotation is not None:
        return annotation, "annotation", dict(recipe=RECIPE, authoritative_poses_preserved=True)
    if sample.modality != "image":
        raise ValueError("Sana demo controls require an image or authoritative pose annotations")
    poses, metadata = demo_camera_trajectory(_camera_path_for_sample(sample) or ["fixed"],
        frame_count=frame_count, suite=sample.suite, repo_root=repo_root)
    probe = probe_image(conditioning_image)
    width, height = int(probe["width"]), int(probe["height"])
    metadata.update(sample_id=sample.sample_id, image_width=width, image_height=height,
                    focal_scale=float(focal_scale))
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    folder = project_root() / "cache/model_runtime/sana_wm" / RECIPE / f"{sample.prediction_stem}--{key}"
    folder.mkdir(parents=True, exist_ok=True)
    intrinsics = _synthetic_intrinsics_sequence(frame_count, image_width=width,
        image_height=height, focal_scale=focal_scale).astype(np.float32)
    _save_verified(folder / "poses.npy", poses)
    _save_verified(folder / "intrinsics.npy", intrinsics)
    metadata.update(poses_sha256=sha(folder / "poses.npy"), intrinsics_sha256=sha(folder / "intrinsics.npy"))
    (folder / "camera_recipe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder, "synthetic_sana_wm_demo_v1", metadata
