from __future__ import annotations

from pathlib import Path

import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.reconstruction import ReconstructionConsistencyMetric
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.reconstruction_consistency_backend import (
    _resolve_depth_anything3_source,
    _score_from_lpips_distances,
)
from worldarena.benchmark.schemas import BenchmarkSample


def _frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


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


def _sample(suite: str = "image_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level=suite,
        modality="image" if suite.startswith("image") else "video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/passageways",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/indoor/passageways/reference.png",
        reference_path="/tmp/reference.png",
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
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        metrics_by_suite={"image_static": ["S_recon", "score_reconstruction"]},
        metric_backends={"reconstruction_consistency": "world_r1_lpips"},
        metric_runtime={"reconstruction_consistency": {"frame_source": "sampled"}},
        normalization={
            "reconstruction_consistency": {
                "lower": 0.0,
                "upper": 1.0,
                "higher_is_better": True,
            }
        },
    )


def test_score_from_lpips_distances_matches_world_r1_s_recon_mapping() -> None:
    result = _score_from_lpips_distances([0.2, 0.4])

    assert result["raw"] == 0.7
    assert result["frame_scores"] == [0.8, 0.6]
    assert result["lpips_mean"] == 0.30000000000000004


def test_depth_anything_source_resolver_supports_world_r1_layout(tmp_path: Path) -> None:
    (tmp_path / "reward_server" / "depth_anything_3").mkdir(parents=True)

    source = _resolve_depth_anything3_source(tmp_path)

    assert source.root == tmp_path.resolve()
    assert source.package_name == "reward_server.depth_anything_3"


def test_reconstruction_consistency_metric_is_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.reconstruction.compute_reconstruction_consistency",
        lambda frames, *, prompt, runtime: {
            "raw": 0.75,
            "backend": "world_r1_lpips",
            "details": {
                "source_metric": "World-R1 S_recon",
                "frame_count": len(frames),
                "prompt": prompt,
            },
            "error": None,
        },
    )

    metric = ReconstructionConsistencyMetric(
        backend="world_r1_lpips",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    media = _media([_frame(), _frame(1)])
    result = metric.compute(_sample("image_static"), media, media)

    assert result.raw == 0.75
    assert result.normalized == 0.75
    assert result.eligibility_status == "diagnostic"
    assert result.backend == "world_r1_lpips"
    assert result.details["source_metric"] == "World-R1 S_recon"
    assert result.details["frame_count"] == 2
    assert result.details["prompt"] == "Move forward without changing the layout."


def test_reconstruction_consistency_requires_video_prediction() -> None:
    metric = ReconstructionConsistencyMetric(
        backend="world_r1_lpips",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    media = _media([_frame()], modality="image")
    result = metric.compute(_sample("image_static"), media, media)

    assert result.raw is None
    assert result.normalized is None
    assert result.eligibility_status == "diagnostic"
    assert "requires a video prediction" in str(result.error)


def test_reconstruction_consistency_registered_with_world_r1_aliases(tmp_path: Path) -> None:
    config = _config(tmp_path)

    metrics = build_metrics(config.metrics_by_suite["image_static"], config)

    assert [metric.name for metric in metrics] == ["reconstruction_consistency"]
    assert METRIC_DIMENSIONS["reconstruction_consistency"] == "consistency_3d_4d"
