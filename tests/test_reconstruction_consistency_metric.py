from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import cv2

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.reconstruction import ReconstructionConsistencyMetric
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.reconstruction_consistency_backend import (
    _photometric_consistency_from_psnr,
    _resolve_depth_anything3_source,
    extract_wbench_frames,
)
from worldarena.benchmark.schemas import BenchmarkSample


def _frame(value: int = 0) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _media(frames: list[np.ndarray], modality: str = "video") -> LoadedMedia:
    return LoadedMedia(
        modality=modality,
        path="/tmp/prediction.mp4",
        frames=frames,
        anchor_frames=frames,
        sampled_frame_indices=list(range(len(frames))),
        protocol_frame_indices=list(range(len(frames))),
        native_frame_count=len(frames),
    )


def _sample(suite: str = "video_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level=suite,
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/passageways",
        path="/tmp/reference.mp4",
        relative_path="static/photorealistic/indoor/passageways/reference.mp4",
        reference_path="/tmp/reference.mp4",
        prediction_stem=f"{suite}-001",
        conditioning_strategy="reference_image",
        prompt_current="A hallway.",
        prompt_target="Move forward without changing the layout.",
        style="photorealistic",
        environment="indoor",
        scene="passageways",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "images",
            video_root=tmp_path / "videos",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            predictions_dir=tmp_path / "predictions",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        metrics_by_suite={"video_static": ["S_recon", "photometric_consistency"]},
        metric_backends={"reconstruction_consistency": "da3_reprojection"},
        metric_runtime={"reconstruction_consistency": {"frame_source": "sampled"}},
        normalization={
            "reconstruction_consistency": {
                "lower": 0.0,
                "upper": 1.0,
                "higher_is_better": True,
            }
        },
    )


def test_photometric_consistency_matches_wbench_mapping() -> None:
    assert _photometric_consistency_from_psnr(20.0) == pytest.approx(1.0 - 10 ** (-1.0))


def test_depth_anything_source_resolver_supports_reward_server_layout(tmp_path: Path) -> None:
    (tmp_path / "reward_server" / "depth_anything_3").mkdir(parents=True)

    source = _resolve_depth_anything3_source(tmp_path)

    assert source.root == tmp_path.resolve()
    assert source.package_name == "reward_server.depth_anything_3"


def test_reconstruction_consistency_metric_is_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.reconstruction.compute_reconstruction_consistency",
        lambda frames, *, prompt, runtime: {
            "raw": 0.75,
            "backend": "da3_reprojection",
            "details": {
                "source_metric": "reconstruction_consistency",
                "geometric_consistency": 0.75,
                "photometric_psnr": 24.5,
                "photometric_consistency": 0.92,
                "frame_count": len(frames),
                "prompt": prompt,
            },
            "error": None,
        },
    )

    metric = ReconstructionConsistencyMetric(
        backend="da3_reprojection",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"frame_source": "sampled"},
    )
    media = _media([_frame(), _frame(1), _frame(2)])
    result = metric.compute(_sample("video_static"), media, media)

    assert result.raw == 0.75
    assert result.normalized == 0.75
    assert result.eligibility_status == "diagnostic"
    assert result.backend == "da3_reprojection"
    assert result.details["geometric_consistency"] == 0.75
    assert result.details["photometric_psnr"] == 24.5
    assert result.details["frame_count"] == 3


def test_reconstruction_consistency_requires_video_prediction() -> None:
    metric = ReconstructionConsistencyMetric(
        backend="da3_reprojection",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"frame_source": "sampled"},
    )
    media = _media([_frame()], modality="image")
    result = metric.compute(_sample("video_static"), media, media)

    assert result.raw is None
    assert result.normalized is None
    assert result.eligibility_status == "diagnostic"
    assert "requires a video prediction" in str(result.error)


def test_reconstruction_consistency_requires_three_frames(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.reconstruction.compute_reconstruction_consistency",
        lambda frames, *, prompt, runtime: {
            "raw": None,
            "backend": "da3_reprojection",
            "details": {"frame_count": len(frames)},
            "error": "reconstruction_consistency requires at least three prediction frames",
        },
    )

    metric = ReconstructionConsistencyMetric(
        backend="da3_reprojection",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"frame_source": "sampled"},
    )
    media = _media([_frame(), _frame(1)])
    result = metric.compute(_sample("video_static"), media, media)

    assert result.raw is None
    assert "at least three prediction frames" in str(result.error)


def test_reconstruction_consistency_registered_with_s_recon_aliases(tmp_path: Path) -> None:
    config = _config(tmp_path)

    metrics = build_metrics(config.metrics_by_suite["video_static"], config)

    assert [metric.name for metric in metrics] == [
        "geometric_consistency",
        "photometric_consistency",
    ]
    assert METRIC_DIMENSIONS["geometric_consistency"] == "consistency_3d_4d"
    assert METRIC_DIMENSIONS["photometric_consistency"] == "consistency_3d_4d"
    assert METRIC_DIMENSIONS["reconstruction_consistency"] == "consistency_3d_4d"


def test_wbench_frame_sampling_matches_official_step_rule(tmp_path: Path) -> None:
    video_path = tmp_path / "sampling.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        (16, 16),
    )
    assert writer.isOpened()
    for index in range(13):
        writer.write(np.full((16, 16, 3), index, dtype=np.uint8))
    writer.release()

    frames, indices, details = extract_wbench_frames(video_path, fps=3.0)
    sequential_frames, sequential_indices, sequential_details = extract_wbench_frames(
        video_path,
        fps=3.0,
        decode_threads=1,
        decode_mode="sequential_exact",
    )

    assert len(frames) == 4
    assert indices == [0, 4, 8, 12]
    assert details["frame_step"] == 4
    assert details["frame_source"] == "wbench_fps"
    assert sequential_indices == indices
    assert all(
        np.array_equal(random_seek, sequential)
        for random_seek, sequential in zip(frames, sequential_frames)
    )
    assert sequential_details["decode_mode"] == "sequential_exact"


def test_wbench_submetrics_share_one_reconstruction_compute(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    def fake_compute(frames, *, prompt, runtime):
        nonlocal calls
        calls += 1
        return {
            "raw": 0.75,
            "backend": "da3_reprojection",
            "details": {
                "geometric_consistency": 0.75,
                "photometric_psnr": 20.0,
                "photometric_consistency": 0.9,
                "frame_count": len(frames),
                "prompt": prompt,
            },
            "error": None,
        }

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.reconstruction.compute_reconstruction_consistency",
        fake_compute,
    )
    metrics = build_metrics(_config(tmp_path).metrics_by_suite["video_static"], _config(tmp_path))
    media = _media([_frame(), _frame(1), _frame(2)])

    outputs = [metric.compute(_sample("video_static"), media, media) for metric in metrics]

    assert calls == 1
    assert [output.raw for output in outputs] == [0.75, 0.9]
    assert [output.eligibility_status for output in outputs] == ["official", "official"]
