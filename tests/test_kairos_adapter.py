from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.kairos import KairosAdapter
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        config_path=tmp_path / "kairos.yaml",
        name="kairos-sensenova",
        family="kairos-sensenova",
        python_bin="python",
        repo_root=tmp_path / "kairos-sensenova",
        checkpoint_dir=tmp_path / "ckpt" / "kairos-sensenova",
        entrypoint="examples/inference.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        generation={
            "variant": "robot-480p-distilled",
            "config_path": "kairos/configs/kairos_4b_config_DMD.py",
            "torchrun_executable": "torchrun",
            "nproc_per_node": 1,
            "run_manage_libs": False,
            "height": 480,
            "width": 640,
            "num_frames": 2,
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


def test_kairos_family_builds_batch_adapter(tmp_path: Path) -> None:
    adapter = build_model_adapter(_make_config(tmp_path))

    assert isinstance(adapter, KairosAdapter)
    assert adapter.supports_batch_generation()


def test_kairos_generate_batch_writes_spec(monkeypatch, tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), color="white").save(reference_path)
    output_path = tmp_path / "predictions" / "image_static_sample_001.mp4"
    config = _make_config(tmp_path)
    adapter = KairosAdapter(config)
    sample = _make_sample(reference_path)
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
                conditioning_image=reference_path,
                output_path=output_path,
                prompt=sample.prompt_target,
            )
        ]
    )

    assert results[sample.sample_id]["status"] == "generated"
    assert captured["label"] == "Kairos Sensenova"
    assert captured["rows"][0]["input_image"] == str(reference_path.resolve())
    assert captured["rows"][0]["prompt"] == sample.prompt_target
    assert captured["generation"] == config.generation
    command = captured["command"]
    assert command[:3] == ["python", "-m", "worldarena.models.adapters.kairos_batch_runner"]
