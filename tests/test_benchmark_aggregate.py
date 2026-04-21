from __future__ import annotations

from worldarena.benchmark import aggregate_results
from worldarena.benchmark.schemas import MetricOutput, SampleScore


def _sample_scores() -> list[SampleScore]:
    return [
        SampleScore(
            sample_id="a",
            suite="image_static",
            split="dev_public",
            metrics={"prompt_alignment": MetricOutput(raw=0.5, normalized=0.5, backend="test")},
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
            metrics={"prompt_alignment": MetricOutput(raw=0.6, normalized=0.6, backend="test")},
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
            metrics={
                "anchor_consistency": MetricOutput(
                    raw=0.7,
                    normalized=0.7,
                    backend="test",
                    eligibility_status="diagnostic",
                )
            },
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
            metrics={"motion_magnitude": MetricOutput(raw=0.8, normalized=0.8, backend="test")},
            path="/tmp/d.avi",
            prediction_path="/tmp/d.avi",
            style="photorealistic",
            environment=None,
            scene=None,
            motion_category="fluid",
        ),
    ]


def test_benchmark_aggregate_separates_official_and_diagnostic_metrics() -> None:
    aggregate = aggregate_results(sample_scores=_sample_scores())

    assert aggregate["per_metric_summary"]["image_static"]["prompt_alignment"] == 0.5
    assert aggregate["per_metric_summary"]["image_dynamic"]["prompt_alignment"] == 0.6
    assert aggregate["per_metric_summary"]["video_dynamic"]["motion_magnitude"] == 0.8
    assert "video_static" not in aggregate["per_metric_summary"]
    assert aggregate["diagnostic_metric_summary"]["video_static"]["anchor_consistency"] == 0.7
    assert aggregate["leaderboard_row"] == {
        "image_dynamic__prompt_alignment": 0.6,
        "image_static__prompt_alignment": 0.5,
        "video_dynamic__motion_magnitude": 0.8,
    }
    assert (
        aggregate["domain_breakdown"]["style"]["photorealistic"]["image_static"][
            "prompt_alignment"
        ]
        == 0.5
    )
    assert "video_static" not in aggregate["domain_breakdown"]["style"]["photorealistic"]
