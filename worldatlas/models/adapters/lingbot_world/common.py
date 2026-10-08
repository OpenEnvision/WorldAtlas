"""Shared camera controls, runtime lifecycle, and launch logic for LingBot-World."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import gc
import hashlib
import json
import numpy as np
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from worldatlas.benchmark.annotations import (
    _matrix_to_pose_vector,
    _pose_vector_to_matrix,
    _resample_pose_vectors,
    load_camera_matrices,
    load_intrinsics_sequence,
    resolve_prompt_contract,
)
from worldatlas.common.annotation_index import split_annotation_reference
from worldatlas.common.checkpoints import (
    project_root as worldatlas_project_root,
    rehome_legacy_workspace_path,
    resolve_checkpoint_path,
    resolve_project_path,
)
from worldatlas.models.adapters.base import ModelAdapter, PreparedGenerationRequest, plan_rollout
from worldatlas.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)
from worldatlas.models.config import ModelRuntimeConfig
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.synthetic_camera import synthetic_camera_matrices


LINGBOT_INTRINSICS_BASE_WIDTH = 832.0
LINGBOT_INTRINSICS_BASE_HEIGHT = 480.0
LINGBOT_DEFAULT_FOCAL_LENGTH = 500.0
LINGBOT_DEFAULT_FRAME_NUM = 81
LINGBOT_OUTPUT_FPS = 16.0
LINGBOT_ROTATION_DEGREES = 30.0
LINGBOT_TILT_DEGREES = 20.0
LINGBOT_ROLL_DEGREES = 15.0
LINGBOT_ORBIT_RADIUS = 1.0
LINGBOT_FORWARD_STEP = 1.0
LINGBOT_LATERAL_STEP = 0.5
LINGBOT_VERTICAL_STEP = 0.35


def _persistent_heartbeat(stage: str, **payload: Any) -> None:
    log_path_value = os.environ.get("LINGBOT_PERSISTENT_HEARTBEAT_LOG")
    if not log_path_value:
        return
    fields: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", "0"),
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        "stage": stage,
    }
    fields.update(payload)
    log_path = Path(log_path_value).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(f"{key}={value}" for key, value in fields.items()) + "\n")
            handle.flush()
    except Exception:
        return


def _project_root(config: ModelRuntimeConfig) -> Path:
    del config
    return worldatlas_project_root()


def _runtime_root(config: ModelRuntimeConfig) -> Path:
    configured = config.generation.get("runtime_cache_dir")
    if configured is not None:
        root = _resolve_extra_path(config, str(configured))
        if root is None:
            raise ValueError("invalid runtime_cache_dir")
    else:
        root = _project_root(config) / "cache" / "model_runtime" / "lingbot_world"
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_extra_path(config: ModelRuntimeConfig, value: str | None) -> Path | None:
    del config
    return resolve_project_path(value)


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _is_materialized(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        return True
    return path.stat().st_size > 0


def _valid_json_file(path: Path) -> bool:
    if not _is_materialized(path):
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def _json_retry_variant(path: Path) -> Path:
    if not path.name.endswith(".json"):
        return path.with_name(f"{path.name}.1")
    return path.with_name(f"{path.name[:-5]}.1.json")


def _repair_diffusers_metadata(overlay_root: Path) -> None:
    subfolders = ("low_noise_model", "high_noise_model")
    metadata_names = ("config.json", "diffusion_pytorch_model.safetensors.index.json")
    for subfolder in subfolders:
        subdir = overlay_root / subfolder
        if not subdir.exists():
            continue
        fallback_subdirs = [overlay_root / sibling for sibling in subfolders if sibling != subfolder]
        for filename in metadata_names:
            target = subdir / filename
            if _valid_json_file(target):
                continue
            candidates = [
                _json_retry_variant(target),
                *(fallback_subdir / filename for fallback_subdir in fallback_subdirs),
                *(_json_retry_variant(fallback_subdir / filename) for fallback_subdir in fallback_subdirs),
            ]
            fallback = next((candidate for candidate in candidates if _valid_json_file(candidate)), None)
            if fallback is not None:
                _ensure_symlink(target, fallback)


def _build_checkpoint_overlay(
    config: ModelRuntimeConfig,
    *,
    overlay_label: str,
    sources: list[Path],
    named_links: dict[str, Path],
) -> Path:
    key = hashlib.sha1(
        "::".join(str(path.resolve()) for path in [*sources, *named_links.values()]).encode("utf-8")
    ).hexdigest()[:12]
    overlay_root = _runtime_root(config) / f"{overlay_label}--{key}"
    overlay_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for child in source.iterdir():
            link_path = overlay_root / child.name
            if link_path.exists() and not _is_materialized(child):
                continue
            _ensure_symlink(link_path, child)
    for name, target in named_links.items():
        _ensure_symlink(overlay_root / name, target)
    _repair_diffusers_metadata(overlay_root)
    return overlay_root


def _prepare_checkpoint_dir(config: ModelRuntimeConfig) -> Path:
    checkpoint_dir = resolve_checkpoint_path(config.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError(f"{config.name} requires checkpoint_dir")

    overlay_sources: list[Path] = []
    shared_checkpoint_value = config.generation.get("shared_checkpoint_dir")
    shared_checkpoint_dir = resolve_checkpoint_path(
        shared_checkpoint_value,
        kind="dir",
        required=shared_checkpoint_value is not None,
    )
    if shared_checkpoint_dir is not None:
        if shared_checkpoint_dir.resolve() != checkpoint_dir:
            overlay_sources.append(shared_checkpoint_dir)
    overlay_sources.append(checkpoint_dir)

    fast_checkpoint_value = config.generation.get("fast_checkpoint_dir")
    fast_checkpoint_dir = resolve_checkpoint_path(
        fast_checkpoint_value,
        kind="dir",
        required=fast_checkpoint_value is not None,
    )
    if fast_checkpoint_dir is None:
        if len(overlay_sources) == 1:
            return checkpoint_dir
        return _build_checkpoint_overlay(
            config,
            overlay_label=checkpoint_dir.name,
            sources=overlay_sources,
            named_links={},
        )

    fast_checkpoint_name = str(config.generation.get("fast_checkpoint_name", "lingbot_world_fast"))
    if len(overlay_sources) == 1 and (checkpoint_dir / fast_checkpoint_name).exists():
        return checkpoint_dir
    return _build_checkpoint_overlay(
        config,
        overlay_label=checkpoint_dir.name,
        sources=overlay_sources,
        named_links={fast_checkpoint_name: fast_checkpoint_dir},
    )


def _normalized_frame_num(frame_num: int) -> int:
    frame_num = max(1, int(frame_num))
    return ((frame_num - 1) // 4) * 4 + 1


def resolve_frame_num(generation: dict[str, Any], default: int = LINGBOT_DEFAULT_FRAME_NUM) -> int:
    """Resolve the output frame count, honoring a target duration when configured."""
    target_seconds = generation.get("target_duration_seconds")
    if target_seconds is None:
        return _normalized_frame_num(int(generation.get("frame_num", default)))
    plan = plan_rollout(
        target_seconds=float(target_seconds),
        native_fps=float(generation.get("output_fps", LINGBOT_OUTPUT_FPS)),
        unit_frames=4,
        base_frames=1,
    )
    return _normalized_frame_num(plan.total_frames)


def _translation_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return matrix


def _rotation_matrix_x(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [cosine, 0.0, -sine],
            [0.0, 1.0, 0.0],
            [sine, 0.0, cosine],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_z(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rigid_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float32)
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def _camera_delta(token: str) -> np.ndarray:
    normalized = str(token or "fixed").strip().lower() or "fixed"
    return synthetic_camera_matrices([normalized], target_frames=2).matrices[-1]


def _camera_path_for_sample(sample: BenchmarkSample) -> list[str]:
    camera_path = [str(token).strip() for token in sample.camera_path if str(token).strip()]
    if camera_path:
        return camera_path
    contract = resolve_prompt_contract(sample.annotation_path)
    fallback = contract.get("camera_path")
    if isinstance(fallback, list):
        normalized = [str(token).strip() for token in fallback if str(token).strip()]
        if normalized:
            return normalized
    if sample.modality == "image":
        return ["fixed"]
    return []


def _interpolate_pose_segment(start_pose: np.ndarray, end_pose: np.ndarray, *, frames: int) -> np.ndarray:
    pose_vectors = np.stack(
        [
            _matrix_to_pose_vector(start_pose.astype(np.float32)),
            _matrix_to_pose_vector(end_pose.astype(np.float32)),
        ],
        axis=0,
    ).astype(np.float32)
    indices = np.asarray([0.0, 1.0], dtype=np.float32)
    interpolated = _resample_pose_vectors(pose_vectors, indices, frames)
    return np.stack([_pose_vector_to_matrix(pose_row) for pose_row in interpolated], axis=0).astype(np.float32)


def _synthetic_image_poses(camera_path: list[str], *, frame_num: int) -> np.ndarray:
    target_frames = _normalized_frame_num(frame_num)
    return synthetic_camera_matrices(
        camera_path,
        target_frames=target_frames,
    ).matrices


def _synthetic_intrinsics_sequence(frame_num: int) -> np.ndarray:
    target_frames = _normalized_frame_num(frame_num)
    intrinsics = np.asarray(
        [
            LINGBOT_DEFAULT_FOCAL_LENGTH,
            LINGBOT_DEFAULT_FOCAL_LENGTH,
            LINGBOT_INTRINSICS_BASE_WIDTH / 2.0,
            LINGBOT_INTRINSICS_BASE_HEIGHT / 2.0,
        ],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None, :], repeats=target_frames, axis=0).astype(np.float32)


def _synthetic_image_action_path(config: ModelRuntimeConfig, sample: BenchmarkSample) -> Path | None:
    if sample.modality != "image":
        return None
    frame_num = resolve_frame_num(config.generation)
    camera_path = _camera_path_for_sample(sample)
    if not camera_path:
        return None

    cache_key_payload = {
        "sample_id": sample.sample_id,
        "relative_path": sample.relative_path,
        "annotation_path": sample.annotation_path,
        "camera_path": camera_path,
        "camera_convention": "worldatlas_diagram_push_positive_z_pull_negative_z",
        "frame_num": frame_num,
    }
    key = hashlib.sha1(
        json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    rank_suffix = ""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        rank_suffix = f"--rank{int(os.environ.get('RANK', '0'))}"
    action_root = _runtime_root(config) / "image_action_paths" / f"{sample.prediction_stem}--{key}{rank_suffix}"
    poses_path = action_root / "poses.npy"
    intrinsics_path = action_root / "intrinsics.npy"
    if poses_path.exists() and intrinsics_path.exists():
        try:
            poses = np.load(poses_path)
            intrinsics = np.load(intrinsics_path)
            if poses.shape == (frame_num, 4, 4) and intrinsics.shape == (frame_num, 4):
                return action_root
        except Exception:
            pass
        shutil.rmtree(action_root, ignore_errors=True)

    action_root.mkdir(parents=True, exist_ok=True)
    poses = _synthetic_image_poses(camera_path, frame_num=frame_num)
    intrinsics = _synthetic_intrinsics_sequence(frame_num)
    np.save(poses_path, poses.astype(np.float32))
    np.save(intrinsics_path, intrinsics.astype(np.float32))
    indexes_text = "\n".join(
        [f"# total {frame_num} indexes", *[f"{index} {index}" for index in range(frame_num)]]
    )
    (action_root / "indexes.txt").write_text(indexes_text + "\n", encoding="utf-8")
    return action_root


def _annotation_action_path(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    *,
    action_string: str | None,
) -> Path | None:
    if "trajectory" in set(config.control_signals):
        synthetic_path = _synthetic_image_action_path(config, sample)
        if synthetic_path is not None:
            return synthetic_path
    if not sample.annotation_path:
        return None
    if split_annotation_reference(sample.annotation_path) is not None:
        return None
    remapped_annotation = rehome_legacy_workspace_path(sample.annotation_path)
    if remapped_annotation is None:
        return None
    candidate = Path(remapped_annotation).expanduser().resolve()
    if not candidate.exists():
        return None

    control_signals = set(config.control_signals)
    intrinsics_path = candidate / "intrinsics.npy"
    poses_path = candidate / "poses.npy"
    action_path = candidate / "action.npy"
    wasd_path = candidate / "wasd_action.npy"
    ijkl_path = candidate / "ijkl_action.npy"
    allow_act2cam = bool(config.generation.get("allow_act2cam", False))

    if "actions" in control_signals:
        if action_string and intrinsics_path.exists():
            return candidate
        if allow_act2cam:
            if intrinsics_path.exists() and wasd_path.exists() and ijkl_path.exists():
                return candidate
        elif intrinsics_path.exists() and poses_path.exists() and action_path.exists():
            return candidate
        if "trajectory" not in control_signals:
            return None

    if "trajectory" in control_signals and intrinsics_path.exists() and poses_path.exists():
        return candidate
    return None


def _intrinsics_are_normalized_for_lingbot(intrinsics: np.ndarray) -> bool:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.ndim != 2 or intrinsics.shape[-1] != 4 or intrinsics.size == 0:
        return False
    if not np.isfinite(intrinsics).all():
        return False
    fx_fy = np.abs(intrinsics[:, :2])
    cx_cy = intrinsics[:, 2:]
    return bool(
        np.max(fx_fy) <= 4.0
        and np.max(cx_cy) <= 2.0
        and np.min(cx_cy) >= -1.0
    )


def _prepare_lingbot_intrinsics(intrinsics: np.ndarray) -> tuple[np.ndarray, bool]:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if not _intrinsics_are_normalized_for_lingbot(intrinsics):
        return intrinsics.astype(np.float32), False

    converted = intrinsics.astype(np.float32).copy()
    converted[:, 0] *= LINGBOT_INTRINSICS_BASE_WIDTH
    converted[:, 2] *= LINGBOT_INTRINSICS_BASE_WIDTH
    converted[:, 1] *= LINGBOT_INTRINSICS_BASE_HEIGHT
    converted[:, 3] *= LINGBOT_INTRINSICS_BASE_HEIGHT
    return converted, True


def _prepare_annotation_action_path(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    annotation_path: Path,
) -> Path:
    pose_matrices = load_camera_matrices(str(annotation_path))
    intrinsics = load_intrinsics_sequence(
        str(annotation_path),
        target_frames=len(pose_matrices) if pose_matrices is not None else None,
    )
    if pose_matrices is None or intrinsics is None:
        return annotation_path
    intrinsics, intrinsics_converted = _prepare_lingbot_intrinsics(intrinsics)

    raw_poses = np.load(annotation_path / "poses.npy")
    raw_intrinsics = np.load(annotation_path / "intrinsics.npy")
    poses_ready = raw_poses.ndim == 3 and raw_poses.shape[-2:] == (4, 4)
    intrinsics_ready = raw_intrinsics.ndim == 2 and raw_intrinsics.shape[-1] == 4
    if poses_ready and intrinsics_ready and not intrinsics_converted:
        return annotation_path

    key = hashlib.sha1(f"{annotation_path.resolve()}::{sample.sample_id}".encode("utf-8")).hexdigest()[:12]
    normalized_dir = _runtime_root(config) / "annotations" / f"{sample.prediction_stem}--{key}"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    np.save(normalized_dir / "poses.npy", pose_matrices.astype(np.float32))
    np.save(normalized_dir / "intrinsics.npy", intrinsics.astype(np.float32))
    for extra_name in ("action.npy", "wasd_action.npy", "ijkl_action.npy", "indexes.txt"):
        source_path = annotation_path / extra_name
        if source_path.exists():
            _ensure_symlink(normalized_dir / extra_name, source_path)
    return normalized_dir


@dataclass(slots=True)
class _LingBotPersistentRuntime:
    config: ModelRuntimeConfig
    torch: Any
    dist: Any | None
    cfg: Any
    generation: dict[str, Any]
    checkpoint_dir: Path
    entrypoint_name: str
    pipeline: Any
    save_video: Any
    max_area_configs: Any
    offload_model: bool
    rank: int
    world_size: int


def _entrypoint_name(config: ModelRuntimeConfig) -> str:
    return Path(str(config.entrypoint or "generate.py")).name


def _is_fast_entrypoint(config: ModelRuntimeConfig) -> bool:
    return _entrypoint_name(config) == "generate_fast.py"


def _resolve_action_controls(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    generation: dict[str, Any],
) -> tuple[str | None, str | None, bool]:
    action_path = generation.get("action_path")
    action_string_by_suite = generation.get("action_string_by_suite", {})
    action_string = action_string_by_suite.get(sample.suite) or generation.get("action_string")
    allow_act2cam = bool(generation.get("allow_act2cam", False))
    if action_path is None:
        resolved_annotation_path = _annotation_action_path(
            config,
            sample,
            action_string=action_string,
        )
        if resolved_annotation_path is not None:
            action_path = str(resolved_annotation_path)
    if not action_path:
        return None, action_string, allow_act2cam

    resolved_action_path = Path(str(action_path)).expanduser()
    if config.repo_root is None:
        raise ValueError("LingBot action resolution requires repo_root")
    if not resolved_action_path.is_absolute():
        resolved_action_path = (config.repo_root / resolved_action_path).resolve()
    if sample.annotation_path:
        remapped_annotation = rehome_legacy_workspace_path(sample.annotation_path)
        annotation_dir = (
            Path(remapped_annotation).expanduser().resolve()
            if remapped_annotation is not None
            else Path(sample.annotation_path).expanduser().resolve()
        )
        if resolved_action_path == annotation_dir:
            resolved_action_path = _prepare_annotation_action_path(
                config,
                sample,
                annotation_dir,
            )
    return str(resolved_action_path), action_string, allow_act2cam


def _validate_single_process_generation_options(
    generation: dict[str, Any],
    *,
    entrypoint_name: str,
) -> None:
    if bool(generation.get("t5_fsdp", False)) or bool(generation.get("dit_fsdp", False)):
        raise ValueError(
            f"{entrypoint_name} batch runner only supports single-process inference; disable t5_fsdp/dit_fsdp"
        )
    if int(generation.get("ulysses_size", 1)) > 1:
        raise ValueError(
            f"{entrypoint_name} batch runner only supports ulysses_size=1 in persistent single-process mode"
        )


def _parse_lingbot_size(value: object) -> tuple[int, int]:
    raw = str(value).lower().replace("x", "*")
    width_text, height_text = raw.split("*", 1)
    return int(width_text), int(height_text)


def _debug_runtime_stage(runtime: _LingBotPersistentRuntime, sample: BenchmarkSample, stage: str, **payload: Any) -> None:
    if os.environ.get("LINGBOT_DEBUG_STAGES") != "1":
        return
    log_path = Path(os.environ.get("LINGBOT_DEBUG_STAGE_LOG", "/tmp/lingbot_debug_stages.log"))
    fields: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rank": runtime.rank,
        "sample_id": sample.sample_id,
        "stage": stage,
    }
    fields.update(payload)
    try:
        if runtime.torch.cuda.is_available():
            fields["mem_alloc_mb"] = round(runtime.torch.cuda.memory_allocated() / 1024 / 1024, 1)
            fields["mem_reserved_mb"] = round(runtime.torch.cuda.memory_reserved() / 1024 / 1024, 1)
    except Exception:
        pass
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(f"{key}={value}" for key, value in fields.items()) + "\n")
        handle.flush()


def _load_batch_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = str(payload["sample_id"])
            results[sample_id] = payload
    return results


def _uses_torchrun_generation(generation: dict[str, Any]) -> bool:
    launcher = str(generation.get("launcher", "")).strip().lower()
    if launcher in {"torchrun", "torch.distributed.run", "distributed"}:
        return True
    return int(generation.get("nproc_per_node", 1)) > 1


def _torchrun_prefix(config: ModelRuntimeConfig, generation: dict[str, Any]) -> list[str]:
    nproc_per_node = int(generation.get("nproc_per_node", generation.get("ulysses_size", 1)))
    command = [
        config.python_bin,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc_per_node),
    ]
    master_addr = generation.get(
        "master_addr",
        os.environ.get("WORLDATLAS_LOCAL_MASTER_ADDR", "127.0.0.1"),
    )
    if master_addr is not None:
        command.extend(["--master_addr", str(master_addr)])
    master_port = os.environ.get(
        "WORLDATLAS_LOCAL_MASTER_PORT",
        generation.get("master_port"),
    )
    if master_port is not None:
        command.extend(["--master_port", str(int(master_port))])
    return command


class LingBotWorldBaseAdapter(ModelAdapter):
    """Shared V1/V2 generation with version-specific pipeline and image handling."""

    @abstractmethod
    def _pipeline_spec(self, wan: Any) -> tuple[Any, dict[str, Any]]:
        """Return the upstream pipeline class and its version-specific options."""

    @staticmethod
    @abstractmethod
    def _generation_kwargs_for_runtime(
        runtime: _LingBotPersistentRuntime,
        *,
        action_path: str | None,
        action_string: str | None,
        allow_act2cam: bool,
    ) -> dict[str, Any]:
        """Build arguments accepted by this version's generate method."""

    @staticmethod
    @abstractmethod
    def _resize_conditioning_image(image: Any, width: int, height: int) -> Any:
        """Apply this version's explicit conditioning_resize policy."""

    def _load_persistent_runtime(
        self,
        *,
        device_id: int = 0,
        distributed: bool = False,
    ) -> _LingBotPersistentRuntime:
        config = self.config
        if config.repo_root is None:
            raise ValueError("LingBot persistent runtime requires repo_root")

        repo_root = config.repo_root.resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        _persistent_heartbeat("runtime_import_start", distributed=distributed, device_id=device_id, repo_root=repo_root)
        import torch
        import torch.distributed as dist
        import wan
        from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        from wan.distributed.util import init_distributed_group
        from wan.utils.utils import save_video
        _persistent_heartbeat("runtime_import_done", distributed=distributed, device_id=device_id)

        generation = dict(config.generation)
        entrypoint_name = _entrypoint_name(config)
        task = str(generation.get("task", "i2v-A14B"))
        cfg = WAN_CONFIGS[task]
        _persistent_heartbeat("checkpoint_prepare_start", task=task)
        checkpoint_dir = _prepare_checkpoint_dir(config)
        _persistent_heartbeat("checkpoint_prepare_done", checkpoint_dir=checkpoint_dir)
        rank = 0
        world_size = 1
        local_device_id = int(device_id)
        if distributed:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_device_id = int(os.environ.get("LOCAL_RANK", local_device_id))
            if world_size > 1:
                _persistent_heartbeat(
                    "dist_init_start",
                    rank=rank,
                    world_size=world_size,
                    local_device_id=local_device_id,
                )
                torch.cuda.set_device(local_device_id)
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        rank=rank,
                        world_size=world_size,
                    )
                _persistent_heartbeat(
                    "dist_init_done",
                    rank=rank,
                    world_size=world_size,
                    local_device_id=local_device_id,
                )
        else:
            _validate_single_process_generation_options(generation, entrypoint_name=entrypoint_name)

        ulysses_size = int(generation.get("ulysses_size", 1))
        if distributed and ulysses_size > 1:
            if ulysses_size != world_size:
                raise ValueError(
                    f"ulysses_size must equal WORLD_SIZE in distributed persistent mode, got "
                    f"{ulysses_size} and {world_size}"
                )
            if cfg.num_heads % ulysses_size != 0:
                raise ValueError(f"{cfg.num_heads=} cannot be divided by {ulysses_size=}")
            _persistent_heartbeat("ulysses_init_start", rank=rank, world_size=world_size, ulysses_size=ulysses_size)
            init_distributed_group()
            _persistent_heartbeat("ulysses_init_done", rank=rank, world_size=world_size, ulysses_size=ulysses_size)

        offload_model = generation.get("offload_model")
        if offload_model is None:
            offload_model = False if distributed and world_size > 1 else True

        torch.cuda.set_device(local_device_id)
        pipeline_cls, pipeline_options = self._pipeline_spec(wan)
        _persistent_heartbeat(
            "pipeline_init_start",
            rank=rank,
            world_size=world_size,
            local_device_id=local_device_id,
            pipeline_cls=pipeline_cls.__name__,
            t5_fsdp=bool(generation.get("t5_fsdp", False)),
            dit_fsdp=bool(generation.get("dit_fsdp", False)),
            use_sp=distributed and ulysses_size > 1,
        )
        pipeline_kwargs: dict[str, Any] = {
            "config": cfg,
            "checkpoint_dir": str(checkpoint_dir),
            "device_id": local_device_id,
            "rank": rank,
            "t5_fsdp": bool(generation.get("t5_fsdp", False)),
            "dit_fsdp": bool(generation.get("dit_fsdp", False)),
            "use_sp": distributed and ulysses_size > 1,
            "t5_cpu": bool(generation.get("t5_cpu", False)),
            "convert_model_dtype": bool(generation.get("convert_model_dtype", False)),
        }
        pipeline_kwargs.update(pipeline_options)
        pipeline = pipeline_cls(**pipeline_kwargs)
        _persistent_heartbeat(
            "pipeline_init_done",
            rank=rank,
            world_size=world_size,
            local_device_id=local_device_id,
            pipeline_cls=pipeline_cls.__name__,
        )
        return _LingBotPersistentRuntime(
            config=config,
            torch=torch,
            dist=dist if distributed else None,
            cfg=cfg,
            generation=generation,
            checkpoint_dir=checkpoint_dir,
            entrypoint_name=entrypoint_name,
            pipeline=pipeline,
            save_video=save_video,
            max_area_configs=MAX_AREA_CONFIGS,
            offload_model=bool(offload_model),
            rank=rank,
            world_size=world_size,
        )

    def _prepare_conditioning_image_for_runtime(self, runtime: _LingBotPersistentRuntime, conditioning_image: Path):
        from PIL import Image

        with Image.open(conditioning_image) as handle:
            image = handle.convert("RGB")
        # LingBot's --size is an area preset; official I2V keeps the reference image
        # aspect ratio. Only resize when an explicit conditioning_resize override is set.
        target_size = runtime.generation.get("conditioning_resize")
        if target_size:
            width, height = _parse_lingbot_size(target_size)
            if image.size != (width, height):
                image = self._resize_conditioning_image(image, width, height)
        return image

    def _run_persistent_generation(
        self,
        runtime: _LingBotPersistentRuntime,
        *,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        conditioning_image = conditioning_image.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        _debug_runtime_stage(runtime, sample, "runtime_enter", conditioning_image=conditioning_image, output_path=output_path)
        action_path, action_string, allow_act2cam = _resolve_action_controls(
            runtime.config,
            sample,
            runtime.generation,
        )
        _debug_runtime_stage(runtime, sample, "action_resolved", action_path=action_path)
        image = self._prepare_conditioning_image_for_runtime(runtime, conditioning_image)
        _debug_runtime_stage(runtime, sample, "image_ready", image_size=image.size)
        generate_kwargs = self._generation_kwargs_for_runtime(
            runtime,
            action_path=action_path,
            action_string=action_string,
            allow_act2cam=allow_act2cam,
        )
        with runtime.torch.no_grad():
            _debug_runtime_stage(runtime, sample, "pipeline_generate_start")
            video = runtime.pipeline.generate(
                prompt,
                image,
                **generate_kwargs,
            )
            _debug_runtime_stage(runtime, sample, "pipeline_generate_done", returned=video is not None)

        if runtime.rank == 0:
            if video is None:
                raise RuntimeError("LingBot rank 0 did not return a generated video tensor")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _debug_runtime_stage(runtime, sample, "save_video_start")
            runtime.save_video(
                tensor=video[None],
                save_file=str(output_path),
                fps=runtime.cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            _debug_runtime_stage(runtime, sample, "save_video_done", exists=output_path.exists())
        del video
        if hasattr(runtime.pipeline, "self_kv_cache"):
            runtime.pipeline.self_kv_cache = None
        _debug_runtime_stage(runtime, sample, "cleanup_start")
        gc.collect()
        runtime.torch.cuda.empty_cache()
        runtime.torch.cuda.ipc_collect()
        _debug_runtime_stage(runtime, sample, "cleanup_done")
        if runtime.dist is not None and runtime.dist.is_initialized():
            _debug_runtime_stage(runtime, sample, "runtime_barrier_start")
            runtime.dist.barrier()
            _debug_runtime_stage(runtime, sample, "runtime_barrier_done")
        if runtime.rank == 0 and not output_path.exists():
            raise FileNotFoundError(f"LingBot output was not written: {output_path}")
        _debug_runtime_stage(runtime, sample, "runtime_return")
        return {
            "prediction_path": str(output_path),
            "prompt": prompt,
            "action_path": action_path,
            "fps": runtime.cfg.sample_fps,
            "entrypoint": runtime.entrypoint_name,
            "checkpoint_dir": str(runtime.checkpoint_dir),
            "persistent_engine": True,
            "checkpoint_load_count": 1,
            "rank": runtime.rank,
            "world_size": runtime.world_size,
        }

    @property
    def distributed_batch_runner_path(self) -> Path:
        """Resolve the runner beside this adapter after package reorganizations."""
        return Path(__file__).with_name("lingbot_world_distributed_batch_runner.py")

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return not bool(self.config.generation.get("disable_batch_generation", False))

    def batch_checkpoint_load_policy(self) -> str:
        """Both LingBot batch runners keep one resident runtime per shard."""
        if not self.supports_batch_generation():
            return "unsupported"
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("LingBot adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("LingBot adapter requires checkpoint_dir")

        project_root = _project_root(self.config)
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("LingBot batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        batch_id = uuid4().hex
        spec_path = spec_dir / f"lingbot_world_{batch_id}.jsonl"
        results_path = spec_dir / f"lingbot_world_{batch_id}.results.jsonl"
        heartbeat_path = spec_dir / f"lingbot_world_{batch_id}.heartbeat.log"
        with spec_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "prediction_stem": request.sample.prediction_stem,
                            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                            "output_path": str(request.output_path.expanduser().resolve()),
                            "prompt": request.prompt,
                            "sample": request.sample.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        generation = dict(self.config.generation)
        device_id = int(generation.get("device_id", generation.get("gpu_index", 0)))
        if _uses_torchrun_generation(generation):
            runner_path = self.distributed_batch_runner_path
            if not runner_path.is_file():
                raise FileNotFoundError(f"LingBot distributed batch runner missing: {runner_path}")
            command = [
                *_torchrun_prefix(self.config, generation),
                str(runner_path),
                "--model-config",
                str(self.config.config_path),
                "--batch-spec-path",
                str(spec_path),
                "--results-path",
                str(results_path),
                "--device-id",
                str(device_id),
                "--heartbeat-path",
                str(heartbeat_path),
            ]
        else:
            command = [
                self.config.python_bin,
                "-m",
                "worldatlas.models.adapters.lingbot_world.lingbot_world_batch_runner",
                "--model-config",
                str(self.config.config_path),
                "--batch-spec-path",
                str(spec_path),
                "--results-path",
                str(results_path),
                "--device-id",
                str(device_id),
            ]
        cli_error: str | None = None
        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        try:
            run_command(command, cwd=project_root, env=env)
        except subprocess.CalledProcessError as exc:
            cli_error = f"LingBot batch runner exited with code {exc.returncode}"

        batch_results = _load_batch_results(results_path)
        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            payload = dict(batch_results.get(request.sample.sample_id, {}))
            if not payload:
                payload = {
                    "status": "failed",
                    "error": cli_error or f"LingBot batch result missing for {request.sample.sample_id}",
                }
            payload.setdefault("command", command)
            payload.setdefault("prediction_path", str(request.output_path))
            payload.setdefault("prompt", request.prompt)
            results[request.sample.sample_id] = payload
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        entrypoint = (self.config.repo_root / self.config.entrypoint).resolve()
        entrypoint_name = _entrypoint_name(self.config)
        checkpoint_dir = _prepare_checkpoint_dir(self.config)
        command = (
            _torchrun_prefix(self.config, generation)
            if _uses_torchrun_generation(generation)
            else [self.config.python_bin]
        )
        command.extend(
            [
            str(entrypoint),
            "--task",
            str(generation.get("task", "i2v-A14B")),
            "--ckpt_dir",
            str(checkpoint_dir),
            "--save_file",
            str(output_path),
            "--prompt",
            prompt,
            "--image",
            str(conditioning_image),
            "--base_seed",
            str(int(generation.get("base_seed", 42))),
            "--size",
            str(generation.get("size", "832*480")),
            "--frame_num",
            str(resolve_frame_num(generation)),
            ]
        )
        if entrypoint_name != "generate_fast.py":
            command.extend(
                [
                    "--sample_solver",
                    str(generation.get("sample_solver", "unipc")),
                ]
            )
        extend_command_with_options(
            command,
            payload=generation,
            value_options=(
                {
                    "sample_shift": float,
                    "ulysses_size": int,
                    "max_attention_size": int,
                }
                if entrypoint_name == "generate_fast.py"
                else {
                    "sample_steps": int,
                    "sample_shift": float,
                    "sample_guide_scale": float,
                    "ulysses_size": int,
                    "prompt_extend_method": str,
                    "prompt_extend_model": str,
                    "prompt_extend_target_lang": str,
                }
            ),
            bool_value_options=("offload_model",),
            flag_options=(
                "convert_model_dtype",
                "t5_fsdp",
                "t5_cpu",
                "dit_fsdp",
                "use_prompt_extend",
            ),
        )

        action_path, action_string, allow_act2cam = _resolve_action_controls(
            self.config,
            sample,
            generation,
        )
        if action_path:
            command.extend(["--action_path", str(action_path)])
        if action_string:
            if entrypoint_name == "generate_fast.py":
                raise ValueError("generate_fast.py does not support action_string or allow_act2cam")
            command.extend(["--action_string", str(action_string), "--allow_act2cam"])
        elif allow_act2cam and action_path:
            if entrypoint_name == "generate_fast.py":
                raise ValueError("generate_fast.py does not support allow_act2cam")
            command.append("--allow_act2cam")

        env = build_env(
            extra_pythonpaths=[self.config.repo_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=self.config.repo_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"LingBot output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }
