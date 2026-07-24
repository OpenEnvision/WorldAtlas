"""Load and transform camera, prompt, and annotation sidecars for benchmark samples."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.common.annotation_index import (
    load_annotation_entry_from_reference,
    split_annotation_reference,
)
from worldarena.common.checkpoints import rehome_legacy_workspace_path


GENERIC_OBJECT_TOKENS = {
    "absence",
    "activity",
    "ambiance",
    "architecture",
    "area",
    "atmosphere",
    "beauty",
    "calmness",
    "camera",
    "city",
    "close-up",
    "closeup",
    "color",
    "combination",
    "description",
    "details",
    "environment",
    "feeling",
    "focus",
    "formation",
    "frame",
    "framing",
    "hues",
    "immersive",
    "immersive quality",
    "life",
    "lighting",
    "motion",
    "movement",
    "nature",
    "objects",
    "perspective",
    "quality",
    "recent activity",
    "scene",
    "setting",
    "shot",
    "sidewalk",
    "soft light",
    "structure",
    "texture",
    "textures",
    "tranquility",
    "video",
    "view",
    "viewer",
    "viewers",
    "vibrant",
}
TEXT_KEYS = ("SceneSummary", "SceneDescription")
STRUCTURED_OBJECT_KEYS = ("objects", "Objects", "entities", "Entities", "subjects", "Subjects")
CAMERA_MOTION_PROMPTS = {
    "push_in": "push in",
    "pull_out": "pull out",
    "move_left": "move left",
    "move_right": "move right",
    "orbit_left": "orbit left",
    "orbit_right": "orbit right",
    "pan_left": "pan left",
    "pan_right": "pan right",
    "tilt_up": "tilt up",
    "tilt_down": "tilt down",
    "roll_cw": "roll clockwise",
    "roll_ccw": "roll counterclockwise",
    "pedestal_up": "pedestal up",
    "pedestal_down": "pedestal down",
    "fixed": "fixed",
}
CAMERA_MOTION_ALIASES = {
    "dolly_in": "push_in",
    "dolly_out": "pull_out",
    "zoom_in": "push_in",
    "zoom_out": "pull_out",
    "truck_left": "move_left",
    "truck_right": "move_right",
    "tiltup": "tilt_up",
    "tiltdown": "tilt_down",
    "rollcw": "roll_cw",
    "rollccw": "roll_ccw",
    "pedestalup": "pedestal_up",
    "pedestaldown": "pedestal_down",
    "stay": "fixed",
}


def _annotation_dir(annotation_path: str | None) -> Path | None:
    if not annotation_path:
        return None
    if split_annotation_reference(annotation_path) is not None:
        return None
    remapped = rehome_legacy_workspace_path(annotation_path)
    if remapped is None:
        return None
    path = Path(remapped)
    return path if path.exists() else None


def _annotation_entry(annotation_path: str | None) -> dict[str, Any]:
    entry = load_annotation_entry_from_reference(annotation_path)
    return entry if isinstance(entry, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _clean_text_list(value: Any) -> list[str]:
    return [text for text in (_clean_text(item) for item in _flatten_strings(value)) if text]


def _normalize_phrase(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    normalized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", normalized)
    if not normalized or normalized in GENERIC_OBJECT_TOKENS:
        return None
    return normalized


def _dedupe_phrases(values: list[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_phrase(value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        flattened: list[str] = []
        for nested in value.values():
            flattened.extend(_flatten_strings(nested))
        return flattened
    if isinstance(value, list):
        flattened: list[str] = []
        for nested in value:
            flattened.extend(_flatten_strings(nested))
        return flattened
    return []


def load_caption(annotation_path: str | None) -> dict[str, Any]:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is not None:
        return _read_json(annotation_dir / "caption.json")
    caption_payload = _annotation_entry(annotation_path).get("caption")
    return caption_payload if isinstance(caption_payload, dict) else {}


def load_instruction_payload(annotation_path: str | None) -> dict[str, Any]:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is not None:
        return _read_json(annotation_dir / "instructions.json")
    instruction_payload = _annotation_entry(annotation_path).get("instructions")
    return instruction_payload if isinstance(instruction_payload, dict) else {}


def load_generation_spec(annotation_path: str | None) -> dict[str, Any]:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is not None:
        return _read_json(annotation_dir / "generation_spec.json")
    generation_spec = _annotation_entry(annotation_path).get("generation_spec")
    return generation_spec if isinstance(generation_spec, dict) else {}


def flatten_instruction_labels(annotation_path: str | None) -> list[str]:
    instructions = load_instruction_payload(annotation_path)
    ordered: list[str] = []
    for _, values in sorted(instructions.items()):
        ordered.extend(_flatten_strings(values))
    return _dedupe_phrases(ordered, limit=16)


def annotation_text(annotation_path: str | None) -> str:
    caption = load_caption(annotation_path)
    parts = [str(caption.get(key, "")).strip() for key in TEXT_KEYS if caption.get(key)]
    return " ".join(part for part in parts if part)


def _normalize_camera_token(value: Any) -> str | None:
    text = _clean_text(value).lower()
    if not text:
        return None
    token = text.replace("-", "_").replace(" ", "_")
    token = CAMERA_MOTION_ALIASES.get(token, token)
    if token in CAMERA_MOTION_PROMPTS:
        return token
    return None


def _camera_segment_prompts(camera_path: list[str], prompt_list: list[str], *, mode: str) -> list[str]:
    prompts: list[str] = []
    if mode == "static":
        for layout, scene_prompt in zip(camera_path, prompt_list[1:]):
            layout_prompt = CAMERA_MOTION_PROMPTS.get(layout, layout.replace("_", " "))
            prompt = f"Camera {layout_prompt}. {scene_prompt}"
            if not prompt.endswith((".", "!", "?")):
                prompt += "."
            prompts.append(prompt)
        return prompts

    if not prompt_list:
        return prompts
    layout = camera_path[0] if camera_path else "fixed"
    layout_prompt = CAMERA_MOTION_PROMPTS.get(layout, layout.replace("_", " "))
    prompt = f"Camera {layout_prompt}. {prompt_list[0]}"
    if not prompt.endswith((".", "!", "?")):
        prompt += "."
    prompts.append(prompt)
    return prompts


def resolve_prompt_contract(annotation_path: str | None) -> dict[str, Any]:
    generation_spec = load_generation_spec(annotation_path)
    if not generation_spec:
        return {
            "mode": None,
            "camera_path": [],
            "prompt_sequence": [],
            "prompt_current": "",
            "prompt_target": "",
        }

    mode = _clean_text(generation_spec.get("mode")).lower() or None
    camera_path = [
        token
        for token in (_normalize_camera_token(item) for item in _flatten_strings(generation_spec.get("camera_path")))
        if token is not None
    ]
    prompt_sequence = _clean_text_list(generation_spec.get("segment_prompts"))
    prompt_list = _clean_text_list(generation_spec.get("prompt_list"))
    prompt_current = _clean_text(generation_spec.get("current_prompt"))
    prompt_target = ""

    if mode == "static":
        if not prompt_current and prompt_list:
            prompt_current = prompt_list[0]
        if not prompt_sequence and camera_path and len(prompt_list) >= len(camera_path) + 1:
            prompt_sequence = _camera_segment_prompts(camera_path, prompt_list, mode="static")
        prompt_target = " ".join(prompt_sequence) if prompt_sequence else (prompt_list[1] if len(prompt_list) > 1 else "")
    elif mode == "dynamic":
        dynamic_prompt = _clean_text(generation_spec.get("prompt"))
        if not prompt_current and prompt_list:
            prompt_current = prompt_list[0]
        if not prompt_current:
            prompt_current = _clean_text(generation_spec.get("current_prompt")) or dynamic_prompt
        if not prompt_sequence and dynamic_prompt:
            prompt_sequence = _camera_segment_prompts(
                camera_path or ["fixed"],
                [dynamic_prompt],
                mode="dynamic",
            )
        prompt_target = prompt_sequence[0] if prompt_sequence else dynamic_prompt
    else:
        if not prompt_current and prompt_list:
            prompt_current = prompt_list[0]
        prompt_target = prompt_sequence[0] if prompt_sequence else (prompt_list[1] if len(prompt_list) > 1 else "")

    if not prompt_current:
        prompt_current = _clean_text(load_caption(annotation_path).get("SceneSummary"))

    return {
        "mode": mode,
        "camera_path": camera_path,
        "prompt_sequence": prompt_sequence,
        "prompt_current": prompt_current,
        "prompt_target": prompt_target,
    }


@lru_cache(maxsize=1)
def _load_spacy_nlp() -> Any | None:
    """Load spacy nlp -> Any | None."""
    try:
        import spacy
    except ModuleNotFoundError:
        return None

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


def _derive_object_prompts_from_text(text: str, limit: int) -> list[str]:
    nlp = _load_spacy_nlp()
    if nlp is None:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        return _dedupe_phrases(tokens, limit=limit)

    doc = nlp(text)
    phrases: list[str] = []
    for chunk in doc.noun_chunks:
        tokens = [
            token.text.lower()
            for token in chunk
            if token.pos_ in {"ADJ", "NOUN", "PROPN"} and not token.is_stop
        ]
        if tokens:
            phrases.append(" ".join(tokens))
    if not phrases:
        phrases = [
            token.text.lower()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop
        ]
    return _dedupe_phrases(phrases, limit=limit)


def derive_object_prompts(annotation_path: str | None, *, limit: int = 6) -> tuple[list[str], str]:
    caption = load_caption(annotation_path)
    generation_spec = load_generation_spec(annotation_path)
    explicit_values: list[str] = []
    explicit_values.extend(_flatten_strings(generation_spec.get("objects")))
    explicit_values.extend(_flatten_strings(generation_spec.get("current_objects")))
    scene_objects = generation_spec.get("scene_objects")
    if isinstance(scene_objects, list) and scene_objects:
        explicit_values.extend(_flatten_strings(scene_objects[0]))
    for key in STRUCTURED_OBJECT_KEYS:
        explicit_values.extend(_flatten_strings(caption.get(key)))
    category_tags = caption.get("CategoryTags")
    if isinstance(category_tags, dict):
        explicit_values.extend(_flatten_strings(category_tags.get("objects")))
        explicit_values.extend(_flatten_strings(category_tags.get("entities")))
    explicit_prompts = _dedupe_phrases(explicit_values, limit=limit)
    if explicit_prompts:
        return explicit_prompts, "annotation_explicit"

    derived = _derive_object_prompts_from_text(annotation_text(annotation_path), limit=limit)
    return derived, "caption_derived"


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion.astype(np.float64)
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = (quaternion / norm).astype(np.float64)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


def _pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    if pose.shape[0] != 7:
        raise ValueError(f"expected pose vector with 7 elements, got shape {pose.shape}")
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quaternion_to_rotation_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3].astype(np.float32)
    return matrix


def _matrix_to_pose_vector(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (4, 4):
        raise ValueError(f"expected camera matrix with shape (4, 4), got {matrix.shape}")
    return np.concatenate(
        [
            matrix[:3, 3].astype(np.float32),
            _rotation_matrix_to_quaternion(matrix[:3, :3]),
        ]
    ).astype(np.float32)


def _load_pose_indices(annotation_dir: Path) -> np.ndarray | None:
    index_path = annotation_dir / "indexes.txt"
    if not index_path.exists():
        return None
    values: list[float] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        values.append(float(parts[1]))
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def _intrinsics_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected 3x3 intrinsics matrix, got {matrix.shape}")
    return np.asarray([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float32)


def _normalize_intrinsics_sequence(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.ndim == 1:
        if intrinsics.shape[0] == 4:
            return intrinsics[None].astype(np.float32)
        raise ValueError(f"unsupported intrinsics vector shape: {intrinsics.shape}")
    if intrinsics.ndim == 2:
        if intrinsics.shape[-1] == 4:
            return intrinsics.astype(np.float32)
        if intrinsics.shape == (3, 3):
            return _intrinsics_matrix_to_vector(intrinsics)[None]
        raise ValueError(f"unsupported intrinsics layout: {intrinsics.shape}")
    if intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        return np.stack([_intrinsics_matrix_to_vector(matrix) for matrix in intrinsics], axis=0)
    raise ValueError(f"unsupported intrinsics layout: {intrinsics.shape}")


def _resample_intrinsics_sequence(
    intrinsics: np.ndarray,
    intrinsics_indices: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if len(intrinsics) == 1:
        return np.repeat(intrinsics.astype(np.float32), target_frames, axis=0)

    target_positions = np.linspace(
        float(intrinsics_indices[0]),
        float(intrinsics_indices[-1]),
        num=target_frames,
        dtype=np.float32,
    )
    columns = [
        np.interp(target_positions, intrinsics_indices, intrinsics[:, idx]).astype(np.float32)
        for idx in range(intrinsics.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def _slerp_quaternion(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    start_unit = start / (np.linalg.norm(start) + 1e-8)
    end_unit = end / (np.linalg.norm(end) + 1e-8)
    dot = float(np.dot(start_unit, end_unit))
    if dot < 0.0:
        end_unit = -end_unit
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = start_unit + ratio * (end_unit - start_unit)
        return (blended / (np.linalg.norm(blended) + 1e-8)).astype(np.float32)
    theta_0 = np.arccos(dot)
    theta = theta_0 * ratio
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    start_scale = np.sin(theta_0 - theta) / sin_theta_0
    end_scale = sin_theta / sin_theta_0
    blended = start_scale * start_unit + end_scale * end_unit
    return (blended / (np.linalg.norm(blended) + 1e-8)).astype(np.float32)


def _resample_pose_vectors(
    pose_vectors: np.ndarray,
    pose_indices: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if len(pose_vectors) == 1:
        return np.repeat(pose_vectors.astype(np.float32), target_frames, axis=0)

    target_positions = np.linspace(
        float(pose_indices[0]),
        float(pose_indices[-1]),
        num=target_frames,
        dtype=np.float32,
    )
    result: list[np.ndarray] = []
    for target_position in target_positions:
        upper = int(np.searchsorted(pose_indices, target_position, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(pose_indices) - 1)
        if lower == upper or pose_indices[upper] == pose_indices[lower]:
            result.append(pose_vectors[lower].astype(np.float32))
            continue
        span = float(pose_indices[upper] - pose_indices[lower])
        ratio = float((target_position - pose_indices[lower]) / span)
        translation = (
            (1.0 - ratio) * pose_vectors[lower, :3] + ratio * pose_vectors[upper, :3]
        ).astype(np.float32)
        rotation = _slerp_quaternion(
            pose_vectors[lower, 3:7],
            pose_vectors[upper, 3:7],
            ratio,
        )
        result.append(np.concatenate([translation, rotation]).astype(np.float32))
    return np.stack(result, axis=0).astype(np.float32)


def load_camera_matrices(
    annotation_path: str | None,
    *,
    target_frames: int | None = None,
) -> np.ndarray | None:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is None:
        return None
    pose_path = annotation_dir / "poses.npy"
    if not pose_path.exists():
        return None
    poses = np.load(pose_path)
    pose_vectors: np.ndarray
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        matrices = poses.astype(np.float32)
        pose_vectors = np.stack([_matrix_to_pose_vector(matrix) for matrix in matrices], axis=0)
    elif poses.ndim == 2 and poses.shape[1] == 7:
        pose_vectors = poses.astype(np.float32)
        matrices = np.stack([_pose_vector_to_matrix(pose_row) for pose_row in pose_vectors], axis=0)
    else:
        raise ValueError(f"unsupported pose array shape: {poses.shape}")

    if target_frames is None or len(matrices) == target_frames:
        return matrices.astype(np.float32)

    pose_indices = _load_pose_indices(annotation_dir)
    if pose_indices is None or len(pose_indices) != len(pose_vectors):
        pose_indices = np.linspace(
            0.0,
            float(len(pose_vectors) - 1),
            num=len(pose_vectors),
            dtype=np.float32,
        )
    interpolated = _resample_pose_vectors(pose_vectors, pose_indices, target_frames)
    return np.stack([_pose_vector_to_matrix(pose_row) for pose_row in interpolated], axis=0)


def load_intrinsics(annotation_path: str | None) -> np.ndarray | None:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is None:
        return None
    intrinsics_path = annotation_dir / "intrinsics.npy"
    if not intrinsics_path.exists():
        return None
    return np.load(intrinsics_path).astype(np.float32)


def load_intrinsics_sequence(
    annotation_path: str | None,
    *,
    target_frames: int | None = None,
) -> np.ndarray | None:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is None:
        return None
    intrinsics_path = annotation_dir / "intrinsics.npy"
    if not intrinsics_path.exists():
        return None

    intrinsics = _normalize_intrinsics_sequence(np.load(intrinsics_path))
    if target_frames is None or len(intrinsics) == target_frames:
        return intrinsics.astype(np.float32)

    intrinsics_indices = _load_pose_indices(annotation_dir)
    if intrinsics_indices is None or len(intrinsics_indices) != len(intrinsics):
        intrinsics_indices = np.linspace(
            0.0,
            float(len(intrinsics) - 1),
            num=len(intrinsics),
            dtype=np.float32,
        )
    return _resample_intrinsics_sequence(intrinsics, intrinsics_indices, target_frames)


__all__ = [
    "annotation_text",
    "derive_object_prompts",
    "flatten_instruction_labels",
    "load_camera_matrices",
    "load_caption",
    "load_generation_spec",
    "load_instruction_payload",
    "load_intrinsics",
    "load_intrinsics_sequence",
    "resolve_prompt_contract",
]
