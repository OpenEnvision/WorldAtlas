from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from worldarena.generation import generate_predictions
from worldarena.models.adapters.matrix_game import _build_action_payload
from worldarena.models.config import ModelRuntimeConfig
from worldarena.benchmark.schemas import BenchmarkSample


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (32, 24))
    assert writer.isOpened()
    for color in ((10, 20, 30), (40, 50, 60), (70, 80, 90)):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :] = color
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _sample(video_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video_static_photorealistic_living_spaces_0001",
        suite="video_static",
        split="dev_public",
        asset_level="video_static",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/living_spaces/0-5s",
        path=str(video_path),
        relative_path="static/photorealistic/indoor/living_spaces/0-5s/0001.avi",
        reference_path=str(video_path),
        prediction_stem="video_static_photorealistic_living_spaces_0001",
        conditioning_strategy="first_frame",
        prompt_current="A photorealistic indoor living spaces scene.",
        prompt_target="Generate a photorealistic video of a coherent indoor living spaces scene.",
        style="photorealistic",
        environment="indoor",
        scene="living_spaces",
        motion_category=None,
        source_name="worldarena_formal_video",
        source_type="curated_video",
        source_group_id="static/photorealistic/indoor/living_spaces::0001",
        license_bucket="C",
        has_annotation=False,
        has_instruction=False,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=32,
        height=24,
        fps=6.0,
        duration_seconds=0.5,
        duration_bucket="0-5s",
    )


def test_generation_runner_extracts_conditioning_frame_and_writes_records(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake_lingbot"
    fake_repo.mkdir(parents=True, exist_ok=True)
    (fake_repo / "generate.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--save_file", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--ckpt_dir")
parser.add_argument("--task")
parser.add_argument("--base_seed")
parser.add_argument("--size")
parser.add_argument("--frame_num")
parser.add_argument("--sample_solver")
args = parser.parse_args()
output = Path(args.save_file)
output.write_bytes(b"fake-video")
Path(str(output) + ".meta.json").write_text(json.dumps({"image": args.image, "prompt": args.prompt}), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "lingbot.yaml",
        name="lingbot-world",
        family="lingbot_world",
        python_bin=sys.executable,
        repo_root=fake_repo,
        checkpoint_dir=tmp_path / "checkpoints",
        entrypoint="generate.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        generation={"task": "i2v-A14B", "frame_num": 9, "size": "320*240", "base_seed": 7},
    )

    result = generate_predictions(
        model_config=model_config,
        manifest=[_sample(video_path)],
        output_dir=tmp_path / "predictions",
        suites=None,
        splits=None,
        limit=None,
        overwrite=True,
    )

    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
    assert output_path.exists()
    meta = json.loads((Path(str(output_path) + ".meta.json")).read_text(encoding="utf-8"))
    assert meta["prompt"] == "Generate a photorealistic video of a coherent indoor living spaces scene."
    assert Path(meta["image"]).exists()
    assert Path(meta["image"]).name == "video_static_photorealistic_living_spaces_0001.png"
    manifest = json.loads((tmp_path / "predictions" / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated"] == 1
    assert result["records"][0]["status"] == "generated"


def test_matrix_game_action_payload_uses_suite_specific_plan() -> None:
    payload = _build_action_payload(
        {
            "action_plan": {
                "default": ["forward"],
                "by_suite": {"video_dynamic": ["forward", "camera_r", "forward_right"]},
                "frames_per_action": 4,
                "first_clip_frames": 12,
                "clip_overlap_frames": 8,
                "num_iterations": 2,
            }
        },
        "video_dynamic",
    )
    assert payload["actions"] == ["forward", "camera_r", "forward_right"]
    assert payload["num_iterations"] == 2
    assert len(payload["keyboard_condition"]) == 60
    assert len(payload["mouse_condition"]) == 60
