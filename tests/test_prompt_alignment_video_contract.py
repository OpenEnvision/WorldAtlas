from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.alignment import PromptAlignmentMetric
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.taxonomy import build_prompts


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image_static_sample",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path="/tmp/reference.png",
        relative_path="reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem="image_static_sample",
        conditioning_strategy="reference_image",
        prompt_current="A photorealistic indoor dining spaces scene.",
        prompt_target="Generate a photorealistic video rollout from the reference image.",
        style="photorealistic",
        environment="indoor",
        scene="dining_spaces",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        has_instruction=True,
    )


def _media(modality: str) -> LoadedMedia:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return LoadedMedia(
        modality=modality,
        path=f"/tmp/prediction.{modality}",
        frames=[frame],
        anchor_frames=[frame],
        evaluation_protocol="segment_random",
        sampled_frame_indices=[0],
        protocol_frame_indices=[0],
        native_frame_count=1,
    )


def test_prompt_alignment_rejects_image_predictions() -> None:
    output = PromptAlignmentMetric(backend="clip", normalization={}).compute(
        sample=_sample(),
        prediction=_media("image"),
        reference=_media("image"),
    )

    assert output.normalized is None
    assert output.eligibility_status == "official"
    assert output.error == "prompt_alignment requires a video prediction"


def test_prompt_alignment_keeps_non_instruction_samples_not_applicable() -> None:
    sample = _sample()
    sample.has_instruction = False

    output = PromptAlignmentMetric(backend="clip", normalization={}).compute(
        sample=sample,
        prediction=_media("image"),
        reference=_media("image"),
    )

    assert output.normalized is None
    assert output.eligibility_status == "not_applicable"


def test_image_suite_default_prompts_request_video_rollouts() -> None:
    _, image_static_target = build_prompts(
        "image_static",
        "photorealistic",
        "indoor",
        "dining_spaces",
        None,
    )
    _, image_dynamic_target = build_prompts(
        "image_dynamic",
        "photorealistic",
        None,
        None,
        "object_motion",
    )

    assert "video rollout" in image_static_target
    assert "video rollout" in image_dynamic_target
    assert "still image" not in image_static_target
    assert "motion motion" not in image_dynamic_target
