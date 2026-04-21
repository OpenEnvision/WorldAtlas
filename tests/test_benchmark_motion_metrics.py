from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.schemas import BenchmarkSample


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video_dynamic_photorealistic_fluid_0001",
        suite="video_dynamic",
        split="dev_public",
        asset_level="video_dynamic",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/videos/fluid",
        path="/tmp/reference.avi",
        relative_path="dynamic/photorealistic/videos/fluid/0001.avi",
        reference_path="/tmp/reference.avi",
        prediction_stem="video_dynamic_photorealistic_fluid_0001",
        conditioning_strategy="first_frame",
        prompt_current="A photorealistic scene with fluid motion.",
        prompt_target="Generate a photorealistic video with fluid motion and coherent appearance.",
        style="photorealistic",
        environment=None,
        scene=None,
        motion_category="fluid",
        source_name="worldarena_formal_video",
        source_type="curated_video",
        source_group_id="dynamic/photorealistic/videos/fluid::0001",
        license_bucket="C",
        has_annotation=False,
        has_instruction=False,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=32,
        height=32,
        fps=6.0,
        duration_seconds=1.0,
        duration_bucket="0-5s",
    )


def _moving_sequence(offset: int) -> list[np.ndarray]:
    positions = [offset, offset + 1, offset + 3, offset + 6]
    frames = []
    for position in positions:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, position : position + 6] = 255
        frames.append(frame)
    return frames


def test_benchmark_motion_metrics_reward_matching_motion_profile() -> None:
    sample = _sample()
    reference = LoadedMedia(
        modality="video",
        path="/tmp/reference.avi",
        frames=_moving_sequence(0),
        anchor_frames=_moving_sequence(0),
    )
    matching_prediction = LoadedMedia(
        modality="video",
        path="/tmp/matching.avi",
        frames=_moving_sequence(0),
        anchor_frames=_moving_sequence(0),
    )
    static_prediction = LoadedMedia(
        modality="video",
        path="/tmp/static.avi",
        frames=[_moving_sequence(0)[0] for _ in range(4)],
        anchor_frames=[_moving_sequence(0)[0] for _ in range(4)],
    )

    magnitude = MotionMagnitudeMetric(
        backend="pixel_diff",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    smoothness = MotionSmoothnessMetric(
        backend="pixel_diff",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )

    matching_mag = magnitude.compute(sample, matching_prediction, reference)
    static_mag = magnitude.compute(sample, static_prediction, reference)
    matching_smooth = smoothness.compute(sample, matching_prediction, reference)
    static_smooth = smoothness.compute(sample, static_prediction, reference)

    assert matching_mag.raw is not None and static_mag.raw is not None
    assert matching_smooth.raw is not None and static_smooth.raw is not None
    assert matching_mag.raw > static_mag.raw
    assert matching_smooth.raw > static_smooth.raw
