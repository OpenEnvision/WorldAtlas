from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from worldarena.models.adapters.longlive_batch_runner import _runtime_config_for_batch
from worldarena.models.config import load_model_config


def test_longlive_runtime_config_preserves_official_defaults(tmp_path: Path) -> None:
    official_config = {
        "model_kwargs": {"model_name": "Wan2.2-TI2V-5B"},
        "data": {
            "data_path": "/path/to/longlive2/inference_prompts",
            "image_or_video_shape": [1, 128, 48, 44, 80],
        },
        "inference": {"sampling_steps": 4, "guidance_scale": 1.0},
        "checkpoints": {"generator_ckpt": "/path/to/longlive2/generator_checkpoint.pt"},
        "output_folder": "videos/longlive2",
        "save_with_index": False,
    }

    runtime_config = _runtime_config_for_batch(
        official_config=official_config,
        rows=[{"prompt": "A test prompt."}],
        prompt_path=tmp_path / "prompts.txt",
        output_folder=tmp_path / "outputs",
        generation={"version": "2.0", "config_overrides": {}},
    )

    assert runtime_config["data"]["image_or_video_shape"] == [1, 128, 48, 44, 80]
    assert runtime_config["inference"]["sampling_steps"] == 4
    assert runtime_config["checkpoints"]["generator_ckpt"] == "/path/to/longlive2/generator_checkpoint.pt"
    assert runtime_config["data"]["data_path"] == str(tmp_path / "prompts.txt")
    assert runtime_config["output_folder"] == str(tmp_path / "outputs")
    assert runtime_config["save_with_index"] is True
    assert runtime_config["inference_iter"] == 0


def test_longlive_runner_dry_run_uses_official_config(tmp_path: Path) -> None:
    config = load_model_config(Path("config/models/longlive_2.yaml"))
    spec_path = tmp_path / "batch.jsonl"
    generation_path = tmp_path / "generation.json"
    image_path = tmp_path / "unused.png"
    output_path = tmp_path / "sample.mp4"
    image_path.write_bytes(b"fake")
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-001",
                "suite": "image_static",
                "prediction_stem": "sample",
                "conditioning_image": str(image_path),
                "output_path": str(output_path),
                "prompt": "A compact silver robot walks through a clean robotics lab.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation_path.write_text(json.dumps(config.generation), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldarena.models.adapters.longlive_batch_runner",
            "--repo_root",
            str(config.repo_root),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
            "--entrypoint",
            str(config.entrypoint),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "WORLDARENA_LONGLIVE_DRY_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["dry_run"] is True
    assert payload["official_config_path"].endswith("configs/inference.yaml")
    assert payload["prompt_path"].endswith("longlive_prompts.txt")
    assert "--config_path" in payload["command"]
