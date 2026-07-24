from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from worldarena.models.adapters.lingbot_video import LingBotVideoAdapter
from worldarena.models.adapters.lingbot_video_batch_runner import (
    frame_count_from_generation,
    structured_prompt_for_row,
)
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def test_lingbot_video_configs_build_registered_adapter() -> None:
    dense = load_model_config(Path("config/models/lingbot_video_dense.yaml"))
    moe = load_model_config(Path("config/models/lingbot_video_moe.yaml"))

    assert isinstance(build_model_adapter(dense), LingBotVideoAdapter)
    assert isinstance(build_model_adapter(moe), LingBotVideoAdapter)
    assert dense.control_signals == ["text", "image"]
    assert dense.reference_mode == "first_frame"
    assert dense.generation["require_persistent_batch"] is True
    assert build_model_adapter(dense).batch_checkpoint_load_policy() == "load_once"
    assert build_model_adapter(moe).batch_checkpoint_load_policy() == "load_once"

    dense_command = build_model_adapter(dense)._build_command(
        Path("batch.jsonl"), Path("generation.json")
    )
    moe_command = build_model_adapter(moe)._build_command(
        Path("batch.jsonl"), Path("generation.json")
    )
    assert "torch.distributed.run" not in dense_command
    assert "torch.distributed.run" in moe_command
    assert "--nproc-per-node=8" in moe_command


def test_structured_prompt_preserves_camera_instruction_and_scene() -> None:
    encoded = structured_prompt_for_row(
        {
            "sample_id": "sample-1",
            "prompt": "Camera orbit left. A lighthouse stands beside a rough sea.",
            "camera_path": ["orbit_left"],
        },
        {"prompt_strategy": "structured"},
    )
    caption = json.loads(encoded)
    description = caption["comprehensive_description"]
    assert description["camera_movement_description"] == "Camera orbit left."
    assert description["scene_content_description"] == (
        "A lighthouse stands beside a rough sea."
    )


def test_lingbot_video_frame_count_matches_official_five_second_default() -> None:
    assert frame_count_from_generation({"duration": 5.0, "fps": 24}) == 121


def test_lingbot_video_batch_runner_dry_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "lingbot-video"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "inference.py").write_text("", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint"
    for component in ("transformer", "text_encoder", "vae"):
        (checkpoint / component).mkdir(parents=True)
    (checkpoint / "model_index.json").write_text("{}", encoding="utf-8")

    spec_path = tmp_path / "batch.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "suite": "image_static",
                "prediction_stem": "sample-1",
                "conditioning_image": str(tmp_path / "input.png"),
                "output_path": str(tmp_path / "output.mp4"),
                "prompt": "Camera fixed. A red robot waits in a workshop.",
                "camera_path": ["fixed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.json"
    generation_path.write_text(
        json.dumps({"duration": 5, "fps": 24, "prompt_strategy": "structured"}),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldarena.models.adapters.lingbot_video_batch_runner",
            "--repo_root",
            str(repo_root),
            "--checkpoint_dir",
            str(checkpoint),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=project_root,
        env={
            **os.environ,
            "PYTHONPATH": str(project_root),
            "WORLDARENA_LINGBOT_VIDEO_DRY_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["dry_run"] is True
    assert payload["persistent_engine"] is True
    assert payload["checkpoint_load_count"] == 1
    assert payload["request_count"] == 1
    assert payload["runtime"]["num_frames"] == 121
