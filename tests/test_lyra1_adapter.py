from __future__ import annotations

import json
from pathlib import Path
import sys

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common import checkpoints
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.lyra1 import Lyra1Adapter
from worldarena.models.adapters.lyra1_batch_runner import _write_stage1_batch_inputs
from worldarena.models.config import ModelRuntimeConfig, load_model_config


def _make_config(tmp_path: Path, *, generation: dict[str, object] | None = None) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "lyra1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    repo_root = tmp_path / "Lyra-1"
    checkpoint_dir = tmp_path / "ckpt" / "Lyra"
    repo_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelRuntimeConfig(
        config_path=config_path,
        name="lyra-1-test",
        family="lyra1",
        python_bin=sys.executable,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        entrypoint="sample.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        generation=dict(generation or {}),
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
        category_path="static/photorealistic/outdoor/aquatic_landscape",
        path=str(reference_path),
        relative_path="static/photorealistic/outdoor/aquatic_landscape/reference.png",
        reference_path=str(reference_path),
        prediction_stem="image_static_sample_001",
        conditioning_strategy="reference_image",
        prompt_current="A coastal landscape.",
        prompt_target="Camera orbit left. A coastal landscape under sunset light.",
        style="photorealistic",
        environment="outdoor",
        scene="aquatic_landscape",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=["orbit_left"],
    )


def _official_memory_generation_options() -> dict[str, object]:
    return {
        "num_gpus": 2,
        "stage1_num_gpus": 1,
        "center_depth_quantile": True,
        "center_depth_quantile_value": 0.9,
        "offload_diffusion_transformer": True,
        "offload_tokenizer": True,
        "offload_text_encoder_model": True,
        "offload_prompt_upsampler": True,
        "offload_guardrail_models": True,
        "disable_guardrail": True,
        "disable_prompt_encoder": True,
        "skip_stage2": True,
    }


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_lyra1_config_uses_worldarena_prompt() -> None:
    config = load_model_config(Path("config/models/lyra1.yaml"))

    assert config.supported_suites == ["image_static"]
    assert config.prompt_mode == "prompt_target"
    assert config.generation["multi_trajectory"] is False


def test_lyra1_stage1_batch_preserves_full_camera_path(tmp_path: Path) -> None:
    batch_input_path = tmp_path / "stage1.jsonl"
    camera_path = ["pan_left", "pull_out", "move_left"]
    _write_stage1_batch_inputs(
        rows=[
            {
                "prediction_stem": "sample",
                "conditioning_image": str(tmp_path / "reference.png"),
                "prompt": "Camera pan left, pull out, then move left.",
                "generation": {
                    "trajectory": "left",
                    "camera_rotation": "trajectory_aligned",
                    "camera_path": camera_path,
                },
            }
        ],
        input_root=tmp_path,
        batch_input_path=batch_input_path,
    )

    row = json.loads(batch_input_path.read_text(encoding="utf-8"))
    assert row["camera_path"] == camera_path
    assert row["trajectory"] == 'worldarena-camera-path:["pan_left","pull_out","move_left"]'


def test_lyra1_checkpoint_layout_accepts_official_and_hf_aliases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from worldarena.models.adapters.lyra1_runner import _checkpoint_layout

    ckpt_root = tmp_path / "ckpt"
    monkeypatch.setenv(checkpoints.CHECKPOINT_ROOT_ENV, str(ckpt_root))
    _touch(ckpt_root / "Lyra" / "lyra_static.pt")
    _touch(ckpt_root / "GEN3C-Cosmos-7B" / "model.pt")
    for name in ("encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt"):
        _touch(ckpt_root / "Cosmos-Tokenize1-CV8x8x8-720p" / name)
    _touch(ckpt_root / "google-t5--t5-11b" / "config.json")
    _touch(ckpt_root / "google-t5--t5-11b" / "spiece.model")
    _touch(ckpt_root / "moge-vitl" / "model.pt")

    layout = _checkpoint_layout(ckpt_root / "Lyra")

    assert layout["gen3c_dir"] == ckpt_root / "GEN3C-Cosmos-7B"
    assert layout["tokenizer_dir"] == ckpt_root / "Cosmos-Tokenize1-CV8x8x8-720p"
    assert layout["t5_dir"] == ckpt_root / "google-t5--t5-11b"
    assert layout["moge_dir"] == ckpt_root / "moge-vitl"


def test_lyra1_generate_command_passes_official_memory_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path / "reference.png")
    output_path = tmp_path / "predictions" / f"{sample.prediction_stem}.mp4"
    adapter = Lyra1Adapter(_make_config(tmp_path, generation=_official_memory_generation_options()))
    captured: dict[str, object] = {}

    def fake_run_command(command, *, cwd, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        output_path.write_bytes(b"video")

    monkeypatch.setattr("worldarena.models.adapters.lyra1.run_command", fake_run_command)

    result = adapter.generate(
        sample=sample,
        conditioning_image=tmp_path / "reference.png",
        output_path=output_path,
        prompt=sample.prompt_target,
    )

    command = captured["command"]
    assert result["prediction_path"] == str(output_path)
    assert "--center_depth_quantile" in command
    assert "--center_depth_quantile_value" in command
    assert "--offload_prompt_upsampler" in command
    assert "--offload_guardrail_models" in command
    assert "--disable_guardrail" in command
    assert "--disable_prompt_encoder" in command


def test_lyra1_batch_command_writes_prompted_spec_and_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path / "reference.png")
    output_path = tmp_path / "predictions" / f"{sample.prediction_stem}.mp4"
    adapter = Lyra1Adapter(_make_config(tmp_path, generation=_official_memory_generation_options()))
    captured: dict[str, object] = {}

    def fake_run_lyra1_command(command, *, cwd, env, log_path):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["log_path"] = log_path
        output_path.write_bytes(b"video")

    monkeypatch.setattr(
        "worldarena.models.adapters.lyra1._run_lyra1_command",
        fake_run_lyra1_command,
    )

    results = adapter.generate_batch(
        [
            PreparedGenerationRequest(
                sample=sample,
                conditioning_image=tmp_path / "reference.png",
                output_path=output_path,
                prompt=sample.prompt_target,
            )
        ]
    )

    command = captured["command"]
    spec_path = Path(command[command.index("--batch_spec_path") + 1])
    rows = [json.loads(line) for line in spec_path.read_text(encoding="utf-8").splitlines()]
    assert results[sample.sample_id]["status"] == "generated"
    assert rows[0]["prompt"] == sample.prompt_target
    assert rows[0]["output_path"] == str(output_path)
    assert "--center_depth_quantile" in command
    assert "--offload_prompt_upsampler" in command
    assert "--offload_guardrail_models" in command
    assert "--disable_guardrail" in command
    assert "--disable_prompt_encoder" in command


def test_lyra1_single_runner_accepts_stage1_memory_flags(monkeypatch, tmp_path: Path) -> None:
    from worldarena.models.adapters import lyra1_runner

    argv = [
        "lyra1_runner",
        "--repo_root",
        str(tmp_path / "Lyra-1"),
        "--checkpoint_dir",
        str(tmp_path / "ckpt" / "Lyra"),
        "--conditioning_image",
        str(tmp_path / "reference.png"),
        "--output_path",
        str(tmp_path / "prediction.mp4"),
        "--sample_name",
        "sample",
        "--prompt",
        "A prompt.",
        "--center_depth_quantile",
        "--center_depth_quantile_value",
        "0.9",
        "--offload_prompt_upsampler",
        "--offload_guardrail_models",
        "--disable_guardrail",
        "--disable_prompt_encoder",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = lyra1_runner.parse_args()
    command = lyra1_runner._stage1_command(
        input_image_path=tmp_path / "reference.png",
        generated_root=tmp_path / "generated",
        args=args,
    )

    assert "--center_depth_quantile" in command
    assert "--center_depth_quantile_value" in command
    assert "--offload_prompt_upsampler" in command
    assert "--offload_guardrail_models" in command
    assert "--disable_guardrail" in command
    assert "--disable_prompt_encoder" in command
