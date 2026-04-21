from __future__ import annotations

from worldarena.benchmark import aggregate_results
from worldarena.benchmark.schemas import MetricOutput, SampleScore


def test_benchmark_aggregate_uses_official_suite_weights() -> None:
    samples = [
        SampleScore(
            sample_id="a",
            suite="image_static",
            split="dev_public",
            score=0.5,
            metrics={"scene_alignment": MetricOutput(raw=0.5, normalized=0.5, backend="test")},
            path="/tmp/a.png",
            prediction_path="/tmp/a.png",
            style="photorealistic",
            environment="indoor",
            scene="living_spaces",
            motion_category=None,
        ),
        SampleScore(
            sample_id="b",
            suite="image_dynamic",
            split="dev_public",
            score=0.6,
            metrics={"dynamic_alignment": MetricOutput(raw=0.6, normalized=0.6, backend="test")},
            path="/tmp/b.png",
            prediction_path="/tmp/b.png",
            style="photorealistic",
            environment=None,
            scene=None,
            motion_category="fluid",
        ),
        SampleScore(
            sample_id="c",
            suite="video_static",
            split="dev_public",
            score=0.7,
            metrics={"rollout_quality": MetricOutput(raw=0.7, normalized=0.7, backend="test")},
            path="/tmp/c.avi",
            prediction_path="/tmp/c.avi",
            style="photorealistic",
            environment="indoor",
            scene="living_spaces",
            motion_category=None,
        ),
        SampleScore(
            sample_id="d",
            suite="video_dynamic",
            split="dev_public",
            score=0.8,
            metrics={"motion_magnitude": MetricOutput(raw=0.8, normalized=0.8, backend="test")},
            path="/tmp/d.avi",
            prediction_path="/tmp/d.avi",
            style="photorealistic",
            environment=None,
            scene=None,
            motion_category="fluid",
        ),
    ]

    aggregate = aggregate_results(
        sample_scores=samples,
        aggregate_weights={
            "image_static": 0.15,
            "image_dynamic": 0.15,
            "video_static": 0.30,
            "video_dynamic": 0.40,
        },
    )

    assert aggregate["overall_scores"]["S_image_static"] == 0.5
    assert aggregate["overall_scores"]["S_video_dynamic"] == 0.8
    assert aggregate["overall_scores"]["S_worldarena"] == 0.695
