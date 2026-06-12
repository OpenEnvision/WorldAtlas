from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.wan22 import Wan22Adapter
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "wan22.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="wan2.2-i2v-a14b",
        family="wan2.2",
        python_bin="python",
        repo_root=tmp_path / "Wan2.2",
        checkpoint_dir=tmp_path / "ckpt" / "Wan2.2-I2V-A14B",
        entrypoint="generate.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_target",
        reference_mode="reference_image",
        generation={
            "task": "i2v-A14B",
            "size": "1280*720",
            "min_seconds": 5.0,
            "offload_model": True,
        },
    )


def _make_sample(reference_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-static-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=str(reference_path),
        relative_path="static/photorealistic/indoor/dining_spaces/reference_001.png",
        reference_path=str(reference_path),
        prediction_stem="image_static_sample_001",
        conditioning_strategy="reference_image",
        prompt_current="A dining room.",
        prompt_target="A dining room with a smooth camera move.",
        style="photorealistic",
        environment="indoor",
        scene="dining_spaces",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


def test_wan22_family_builds_batch_adapter(tmp_path: Path) -> None:
    adapter = build_model_adapter(_make_config(tmp_path))

    assert isinstance(adapter, Wan22Adapter)
    assert adapter.supports_batch_generation()


def test_wan22_generate_batch_writes_spec_and_checks_outputs(monkeypatch, tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), color="white").save(reference_path)
    output_path = tmp_path / "predictions" / "image_static_sample_001.mp4"
    config = _make_config(tmp_path)
    config.repo_root.mkdir(parents=True)
    config.checkpoint_dir.mkdir(parents=True)
    adapter = Wan22Adapter(config)
    sample = _make_sample(reference_path)
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, log_path):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["log_path"] = log_path
        spec_path = Path(command[command.index("--batch_spec_path") + 1])
        rows = [
            json.loads(line)
            for line in spec_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        captured["rows"] = rows
        for row in rows:
            Path(row["output_path"]).write_bytes(b"video")

    monkeypatch.setattr("worldarena.models.adapters.wan22._run_wan22_command", fake_run)

    results = adapter.generate_batch(
        [
            PreparedGenerationRequest(
                sample=sample,
                conditioning_image=reference_path,
                output_path=output_path,
                prompt=sample.prompt_target,
            )
        ]
    )

    assert results[sample.sample_id]["status"] == "generated"
    assert output_path.read_bytes() == b"video"
    assert captured["rows"] == [
        {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_stem": sample.prediction_stem,
            "conditioning_image": str(reference_path.resolve()),
            "output_path": str(output_path.resolve()),
            "prompt": sample.prompt_target,
        }
    ]
    command = captured["command"]
    assert command[:3] == ["python", "-m", "worldarena.models.adapters.wan22_batch_runner"]
    assert "--offload_model" in command
