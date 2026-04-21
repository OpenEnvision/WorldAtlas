from __future__ import annotations

from worldarena.benchmark import aggregate_results
from worldarena.benchmark.config import BenchmarkAggregationConfig
from worldarena.benchmark.schemas import MetricOutput, SampleScore


def _sample_scores() -> list[SampleScore]:
    return [
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


def test_benchmark_aggregate_defaults_to_official_suite_scores_only() -> None:
    aggregate = aggregate_results(sample_scores=_sample_scores())

    assert aggregate["overall_scores"]["S_image_static"] == 0.5
    assert aggregate["overall_scores"]["S_image_dynamic"] == 0.6
    assert aggregate["overall_scores"]["S_video_static"] == 0.7
    assert aggregate["overall_scores"]["S_video_dynamic"] == 0.8
    assert "S_worldarena_derived" not in aggregate["overall_scores"]
    assert aggregate["leaderboard_row"] == aggregate["overall_scores"]


def test_benchmark_aggregate_optionally_emits_derived_world_score() -> None:
    aggregate = aggregate_results(
        sample_scores=_sample_scores(),
        aggregation=BenchmarkAggregationConfig(enabled=True),
    )

    assert aggregate["overall_scores"]["S_worldarena_derived"] == 0.695
    assert aggregate["leaderboard_row"]["S_worldarena_derived"] == 0.695
