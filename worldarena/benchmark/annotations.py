from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


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


def _annotation_dir(annotation_path: str | None) -> Path | None:
    if not annotation_path:
        return None
    path = Path(annotation_path)
    return path if path.exists() else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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
    if annotation_dir is None:
        return {}
    return _read_json(annotation_dir / "caption.json")


def load_instruction_payload(annotation_path: str | None) -> dict[str, Any]:
    annotation_dir = _annotation_dir(annotation_path)
    if annotation_dir is None:
        return {}
    return _read_json(annotation_dir / "instructions.json")


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


@lru_cache(maxsize=1)
def _load_spacy_nlp() -> Any | None:
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
    explicit_values: list[str] = []
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


__all__ = [
    "annotation_text",
    "derive_object_prompts",
    "flatten_instruction_labels",
    "load_camera_matrices",
    "load_caption",
    "load_instruction_payload",
    "load_intrinsics",
]
