from __future__ import annotations

from pathlib import Path

import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.gen3c import (
    Gen3CAdapter,
    _generation_for_sample,
)
from worldarena.models.config import ModelRuntimeConfig


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "gen3c.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="gen3c",
        family="gen3c",
        python_bin="python",
        repo_root=tmp_path,
        checkpoint_dir=tmp_path,
        entrypoint="cosmos_predict1/diffusion/inference/gen3c_single_image.py",
        output_ext=".mp4",
        supported_suites=["image_static", "video_static", "video_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
    )


def _make_image_static_sample(reference_path: Path, *, camera_path: list[str]) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-static-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/aquatic_landscape",
        path=str(reference_path),
        relative_path="static/photorealistic/outdoor/aquatic_landscape/reference.png",
        reference_path=str(reference_path),
        prediction_stem="image_static_sample_001",
        conditioning_strategy="reference_image",
        prompt_current="A coastal landscape under a sunset sky.",
        prompt_target="Camera orbit left. A coastal landscape with ocean waves and distant hills.",
        style="photorealistic",
        environment="outdoor",
        scene="aquatic_landscape",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=camera_path,
        generation_mode="static",
    )


def test_gen3c_image_static_uses_prompt_target(tmp_path: Path) -> None:
    sample = _make_image_static_sample(tmp_path / "reference.png", camera_path=["orbit_left"])
    adapter = Gen3CAdapter(_make_config(tmp_path))

    assert adapter.prompt_for(sample) == sample.prompt_target


@pytest.mark.parametrize(
    ("camera_token", "trajectory", "camera_rotation"),
    [
        ("orbit_left", "clockwise", "center_facing"),
        ("orbit_right", "counterclockwise", "center_facing"),
        ("push_in", "zoom_in", "center_facing"),
        ("pull_out", "zoom_out", "center_facing"),
        ("move_left", "left", "no_rotation"),
        ("pan_left", "left", "trajectory_aligned"),
        ("pan_right", "right", "trajectory_aligned"),
    ],
)
def test_gen3c_image_static_maps_camera_path_to_trajectory(
    tmp_path: Path,
    camera_token: str,
    trajectory: str,
    camera_rotation: str,
) -> None:
    sample = _make_image_static_sample(tmp_path / "reference.png", camera_path=[camera_token])

    generation = _generation_for_sample({"trajectory": "left", "camera_rotation": "center_facing"}, sample)

    assert generation["trajectory"] == trajectory
    assert generation["camera_rotation"] == camera_rotation
    assert generation["camera_path"] == [camera_token]
    assert generation["camera_path_primary"] == camera_token


def test_gen3c_empty_prompt_still_fails_through_base_contract(tmp_path: Path) -> None:
    sample = _make_image_static_sample(tmp_path / "reference.png", camera_path=["orbit_left"])
    sample.prompt_target = ""
    sample.prompt_current = ""
    adapter = Gen3CAdapter(_make_config(tmp_path))

    with pytest.raises(ValueError, match="requires non-empty prompt_target"):
        adapter.prompt_for(sample)
