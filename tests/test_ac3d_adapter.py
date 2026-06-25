from __future__ import annotations

import json
from pathlib import Path

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.ac3d import AC3DAdapter
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.config import ModelRuntimeConfig, load_model_config
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "ac3d.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="ac3d-cogvideox-5b-i2v",
        family="ac3d",
        python_bin="python",
        repo_root=tmp_path / "ac3d",
        checkpoint_dir=tmp_path / "ckpt" / "ac3d",
        entrypoint="inference/cli_demo_camera.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_target",
        reference_mode="reference_image",
        control_signals=["text", "image", "camera_pose"],
        generation={
            "variant": "5b",
            "conditioning_mode": "image",
            "video_root_dir": str(tmp_path / "re10k"),
            "annotation_json": "annotations/test.json",
            "base_model_path": str(tmp_path / "ckpt" / "CogVideoX-5b-I2V"),
            "controlnet_model_path": "5B/checkpoint-10000.pt",
            "camera_index": 3,
            "num_inference_steps": 1,
            "num_frames": 49,
            "fps": 8,
            "min_duration_seconds": 5.0,
        },
    )


def _make_sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-static-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/indoor/dining_spaces/reference_001.png",
        reference_path="/tmp/reference.png",
        prediction_stem="image_static_sample_001",
        conditioning_strategy="none",
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
        camera_path=["orbit_left"],
    )


def test_ac3d_config_and_registry_load() -> None:
    config = load_model_config(Path("config/models/ac3d.yaml"))
    adapter = build_model_adapter(config)

    assert isinstance(adapter, AC3DAdapter)
    assert adapter.supports_batch_generation()
    assert config.reference_mode == "reference_image"
    assert config.generation["variant"] == "5b"
    assert config.generation["controlnet_transformer_attention_head_dim"] == 64


def test_ac3d_generate_batch_writes_spec_and_generation_config(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "predictions" / "image_static_sample_001.mp4"
    config = _make_config(tmp_path)
    adapter = AC3DAdapter(config)
    sample = _make_sample()
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, env, log_path, label):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["log_path"] = log_path
        captured["label"] = label
        spec_path = Path(command[command.index("--batch_spec_path") + 1])
        generation_path = Path(command[command.index("--generation_config_path") + 1])
        captured["rows"] = [
            json.loads(line)
            for line in spec_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        captured["generation"] = json.loads(generation_path.read_text(encoding="utf-8"))
        for row in captured["rows"]:
            Path(row["output_path"]).write_bytes(b"video")

    monkeypatch.setattr("worldarena.models.adapters.external_batch.run_logged_command", fake_run)

    results = adapter.generate_batch(
        [
            PreparedGenerationRequest(
                sample=sample,
                conditioning_image=tmp_path / "unused",
                output_path=output_path,
                prompt=sample.prompt_target,
            )
        ]
    )

    assert results[sample.sample_id]["status"] == "generated"
    assert captured["label"] == "AC3D"
    assert captured["rows"] == [
        {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_stem": sample.prediction_stem,
            "conditioning_image": str((tmp_path / "unused").resolve()),
            "output_path": str(output_path.resolve()),
            "prompt": sample.prompt_target,
            "camera_index": 3,
            "start_camera_idx": 3,
            "end_camera_idx": 4,
            "camera_path": ["orbit_left"],
        }
    ]
    assert captured["generation"] == config.generation
    command = captured["command"]
    assert command[:3] == ["python", "-m", "worldarena.models.adapters.ac3d_batch_runner"]
