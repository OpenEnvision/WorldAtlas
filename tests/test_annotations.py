from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldarena.benchmark.annotations import (
    derive_object_prompts,
    flatten_instruction_labels,
    load_camera_matrices,
)


def _write_annotation_dir(base_dir: Path) -> Path:
    annotation_dir = base_dir / "sample_ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "caption.json").write_text(
        json.dumps(
            {
                "SceneSummary": "A coral reef with fish and seaweed.",
                "SceneDescription": "Bright coral and several tropical fish move through the reef.",
            }
        ),
        encoding="utf-8",
    )
    (annotation_dir / "instructions.json").write_text(
        json.dumps({"0->12": ["dolly in", "pan left"]}),
        encoding="utf-8",
    )
    return annotation_dir


def test_load_camera_matrices_supports_pose_vectors(tmp_path: Path) -> None:
    annotation_dir = _write_annotation_dir(tmp_path)
    poses = np.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=np.float32,
    )
    np.save(annotation_dir / "poses.npy", poses)

    matrices = load_camera_matrices(str(annotation_dir))

    assert matrices is not None
    assert matrices.shape == (2, 4, 4)
    np.testing.assert_allclose(matrices[0, :3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(matrices[0, :3, :3], np.eye(3, dtype=np.float32), atol=1e-5)


def test_load_camera_matrices_interpolates_to_target_frames(tmp_path: Path) -> None:
    annotation_dir = _write_annotation_dir(tmp_path)
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(annotation_dir / "poses.npy", poses)
    (annotation_dir / "indexes.txt").write_text("0 0\n1 10\n", encoding="utf-8")

    matrices = load_camera_matrices(str(annotation_dir), target_frames=6)

    assert matrices is not None
    assert matrices.shape == (6, 4, 4)
    np.testing.assert_allclose(
        matrices[:, 0, 3],
        np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32),
        atol=1e-4,
    )
    expected_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 6, axis=0)
    np.testing.assert_allclose(matrices[:, :3, :3], expected_rotations, atol=1e-5)


def test_derive_object_prompts_falls_back_to_caption_text(tmp_path: Path) -> None:
    annotation_dir = _write_annotation_dir(tmp_path)

    prompts, prompt_source = derive_object_prompts(str(annotation_dir))

    assert prompt_source == "caption_derived"
    assert prompts


def test_derive_object_prompts_prefers_explicit_annotation_fields(tmp_path: Path) -> None:
    annotation_dir = _write_annotation_dir(tmp_path)
    (annotation_dir / "caption.json").write_text(
        json.dumps(
            {
                "SceneSummary": "A coral reef with fish and seaweed.",
                "objects": ["coral reef", "fish", "seaweed"],
            }
        ),
        encoding="utf-8",
    )

    prompts, prompt_source = derive_object_prompts(str(annotation_dir))

    assert prompt_source == "annotation_explicit"
    assert prompts == ["coral reef", "fish", "seaweed"]


def test_flatten_instruction_labels_preserves_unique_labels(tmp_path: Path) -> None:
    annotation_dir = _write_annotation_dir(tmp_path)

    labels = flatten_instruction_labels(str(annotation_dir))

    assert labels == ["dolly in", "pan left"]
