from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldarena.benchmark import long_sequence_native


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_compute_long_aesthetic_quality_native_aggregates_clip_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    clip_a = tmp_path / "split_clip" / "desert" / "desert_000.mp4"
    clip_b = tmp_path / "split_clip" / "desert" / "desert_001.mp4"
    clip_a.parent.mkdir(parents=True)
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "prompt_en": "desert",
                    "dimension": ["aesthetic_quality"],
                    "video_list": [str(clip_a), str(clip_b)],
                }
            ]
        ),
        encoding="utf-8",
    )

    frame_map = {
        str(clip_a): [_frame(2), _frame(2)],
        str(clip_b): [_frame(8), _frame(8)],
    }

    monkeypatch.setattr(
        long_sequence_native,
        "read_video_frames",
        lambda path: frame_map[str(path)],
    )
    monkeypatch.setattr(
        long_sequence_native,
        "compute_quality_component_scores",
        lambda frames, *, component_name: {
            "backend": "laion_aes",
            "raw": float(frames[0][0, 0, 0]) / 10.0,
            "frame_scores": [float(frames[0][0, 0, 0]) / 10.0 for _ in frames],
            "native_frame_scores": [float(frames[0][0, 0, 0]) for _ in frames],
            "native_score_range": [0.0, 10.0],
            "component_name": component_name,
        },
    )

    raw, clip_results, video_results = long_sequence_native.compute_long_aesthetic_quality(
        str(manifest_path),
        "cpu",
        None,
    )

    assert raw == 0.5
    assert [item["video_results"] for item in clip_results] == [0.2, 0.8]
    assert clip_results[0]["quality_backend"] == "laion_aes"
    assert len(video_results) == 1
    assert video_results[0]["video_results"] == 0.5
    assert video_results[0]["clip_scores"] == [0.2, 0.8]


def test_compute_long_imaging_quality_native_uses_musiq_backend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    clip_a = tmp_path / "split_clip" / "city" / "city_000.mp4"
    clip_b = tmp_path / "split_clip" / "city" / "city_001.mp4"
    clip_a.parent.mkdir(parents=True)
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "prompt_en": "city",
                    "dimension": ["imaging_quality"],
                    "video_list": [str(clip_a), str(clip_b)],
                }
            ]
        ),
        encoding="utf-8",
    )

    frame_map = {
        str(clip_a): [_frame(40)],
        str(clip_b): [_frame(60)],
    }
    calls: list[tuple[str | None, str]] = []

    monkeypatch.setattr(
        long_sequence_native,
        "read_video_frames",
        lambda path: frame_map[str(path)],
    )
    monkeypatch.setattr(
        long_sequence_native,
        "compute_musiq_scores",
        lambda frames, *, model_path, preprocess_mode: (
            calls.append((model_path, preprocess_mode))
            or {
                "backend": "musiq",
                "raw": float(frames[0][0, 0, 0]) / 100.0,
                "frame_scores": [float(frames[0][0, 0, 0]) / 100.0 for _ in frames],
                "native_frame_scores": [float(frames[0][0, 0, 0]) for _ in frames],
                "native_score_range": [0.0, 100.0],
                "preprocess_mode": preprocess_mode,
                "weights_path": "/tmp/musiq.pth",
            }
        ),
    )

    raw, clip_results, video_results = long_sequence_native.compute_long_imaging_quality(
        str(manifest_path),
        "cpu",
        {"model_path": "/tmp/native-musiq.pth"},
        imaging_quality_preprocessing_mode="shorter",
    )

    assert raw == 0.5
    assert calls == [
        ("/tmp/native-musiq.pth", "shorter"),
        ("/tmp/native-musiq.pth", "shorter"),
    ]
    assert [item["video_results"] for item in clip_results] == [0.4, 0.6]
    assert clip_results[0]["quality_backend"] == "musiq"
    assert video_results[0]["video_results"] == 0.5
    assert video_results[0]["quality_backends"] == ["musiq"]


def test_compute_long_motion_smoothness_native_aggregates_clip_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    clip_a = tmp_path / "split_clip" / "rain" / "rain_000.mp4"
    clip_b = tmp_path / "split_clip" / "rain" / "rain_001.mp4"
    clip_a.parent.mkdir(parents=True)
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "prompt_en": "rain",
                    "dimension": ["motion_smoothness"],
                    "video_list": [str(clip_a), str(clip_b)],
                }
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        long_sequence_native,
        "compute_amt_motion_smoothness",
        lambda clip_path, *, device, config_path, ckpt_path: (
            calls.append((clip_path, device))
            or {
                "backend": "amt_s",
                "raw": 0.25 if clip_path.endswith("000.mp4") else 0.75,
                "vfi_difference": 42.0,
            }
        ),
    )

    raw, clip_results, video_results = long_sequence_native.compute_long_motion_smoothness(
        str(manifest_path),
        "cpu",
        None,
    )

    assert raw == 0.5
    assert calls == [(str(clip_a), "cpu"), (str(clip_b), "cpu")]
    assert [item["video_results"] for item in clip_results] == [0.25, 0.75]
    assert video_results[0]["video_results"] == 0.5
    assert video_results[0]["backends"] == ["amt_s"]


def test_compute_long_dynamic_degree_native_aggregates_clip_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    clip_a = tmp_path / "split_clip" / "street" / "street_000.mp4"
    clip_b = tmp_path / "split_clip" / "street" / "street_001.mp4"
    clip_a.parent.mkdir(parents=True)
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "prompt_en": "street",
                    "dimension": ["dynamic_degree"],
                    "video_list": [str(clip_a), str(clip_b)],
                }
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        long_sequence_native,
        "compute_raft_dynamic_degree",
        lambda clip_path, *, device, model_path: (
            calls.append((clip_path, model_path))
            or {
                "backend": "raft_things",
                "raw": 0.0 if clip_path.endswith("000.mp4") else 1.0,
                "active_transition_count": 3,
            }
        ),
    )

    raw, clip_results, video_results = long_sequence_native.compute_long_dynamic_degree(
        str(manifest_path),
        "cpu",
        {"model": "/tmp/raft-things.pth"},
    )

    assert raw == 0.5
    assert calls == [
        (str(clip_a), "/tmp/raft-things.pth"),
        (str(clip_b), "/tmp/raft-things.pth"),
    ]
    assert [item["video_results"] for item in clip_results] == [0.0, 1.0]
    assert video_results[0]["video_results"] == 0.5
    assert video_results[0]["backends"] == ["raft_things"]


def test_long_sequence_runtime_specs_route_native_dimensions_to_worldarena() -> None:
    motion_module = long_sequence_native.compute_long_motion_smoothness.__module__
    dynamic_module = long_sequence_native.compute_long_dynamic_degree.__module__
    aesthetic_module = long_sequence_native.compute_long_aesthetic_quality.__module__
    imaging_module = long_sequence_native.compute_long_imaging_quality.__module__

    from worldarena.benchmark.long_sequence_dispatch import LONG_SEQUENCE_EVALUATOR_SPECS

    assert LONG_SEQUENCE_EVALUATOR_SPECS["motion_smoothness"][0] == motion_module
    assert LONG_SEQUENCE_EVALUATOR_SPECS["dynamic_degree"][0] == dynamic_module
    assert LONG_SEQUENCE_EVALUATOR_SPECS["aesthetic_quality"][0] == aesthetic_module
    assert LONG_SEQUENCE_EVALUATOR_SPECS["imaging_quality"][0] == imaging_module
