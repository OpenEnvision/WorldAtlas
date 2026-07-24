from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices
from worldarena.models.adapters.cami2v import CamI2VAdapter
from worldarena.models.adapters.cami2v_batch_runner import (
    OFFICIAL_PROMPT_SUFFIX,
    _disable_demo_prompt_suffix,
)
from worldarena.models.adapters.camera_i2v_common import write_official_camera_file
from worldarena.models.adapters.realcam_i2v import RealCamI2VAdapter
from worldarena.models.adapters.realcam_i2v_batch_runner import (
    _install_inference_models_namespace,
)
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def test_camera_i2v_configs_build_registered_batch_adapters() -> None:
    expected = {
        "config/models/cami2v.yaml": CamI2VAdapter,
        "config/models/realcam_i2v.yaml": RealCamI2VAdapter,
    }
    for config_path, adapter_type in expected.items():
        config = load_model_config(Path(config_path))
        adapter = build_model_adapter(config)
        assert isinstance(adapter, adapter_type)
        assert adapter.supports_batch_generation()
        assert config.supported_suites == ["image_static", "image_dynamic"]
        assert config.control_signals == ["text", "image", "camera_pose"]


def test_official_camera_file_preserves_worldarena_multisegment_c2w(tmp_path: Path) -> None:
    camera_path = ["push_in", "pan_left", "pedestal_up"]
    target_frames = 17
    camera_file = tmp_path / "camera.txt"

    normalized = write_official_camera_file(
        camera_file,
        camera_path,
        target_frames=target_frames,
    )

    assert normalized == tuple(camera_path)
    rows = np.loadtxt(camera_file, comments="https")
    assert rows.shape == (target_frames, 19)
    w2cs_3x4 = rows[:, 7:].reshape(-1, 3, 4)
    w2cs = np.repeat(np.eye(4, dtype=np.float64)[None], repeats=target_frames, axis=0)
    w2cs[:, :3, :] = w2cs_3x4
    reconstructed_c2ws = np.linalg.inv(w2cs)
    expected_c2ws = synthetic_camera_matrices(
        camera_path,
        target_frames=target_frames,
    ).matrices
    np.testing.assert_allclose(reconstructed_c2ws, expected_c2ws, atol=1e-6)


def test_cami2v_demo_prompt_suffix_is_removed_at_processor_boundary() -> None:
    class FakeProcessor:
        def get_batch_input(self, ref_img, caption, *args, **kwargs):
            return caption

    module = SimpleNamespace(SingleImageForInference=FakeProcessor)
    _disable_demo_prompt_suffix(module)
    processor = FakeProcessor()

    assert processor.get_batch_input(None, "A coastal scene." + OFFICIAL_PROMPT_SUFFIX) == (
        "A coastal scene."
    )
    assert processor.get_batch_input(None, "A prompt without the suffix") == (
        "A prompt without the suffix"
    )


def test_realcam_inference_namespace_skips_eager_training_import(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    camera_root = models_root / "camera_controller"
    camera_root.mkdir(parents=True)
    (models_root / "__init__.py").write_text(
        "raise AssertionError('training package initializer must not execute')\n",
        encoding="utf-8",
    )
    (camera_root / "probe.py").write_text("VALUE = 17\n", encoding="utf-8")

    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "models" or name.startswith("models.")
    }
    try:
        _install_inference_models_namespace(tmp_path)
        probe = importlib.import_module("models.camera_controller.probe")
        assert probe.VALUE == 17
    finally:
        for name in tuple(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        sys.modules.update(previous)


def _run_dry_runner(
    *,
    tmp_path: Path,
    module: str,
    env_name: str,
    config_path: str,
) -> dict[str, object]:
    config = load_model_config(Path(config_path))
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"not decoded during dry-run")
    output_path = tmp_path / "output.mp4"
    spec_path = tmp_path / f"{module.rsplit('.', 1)[-1]}.jsonl"
    generation_path = spec_path.with_suffix(".generation.json")
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-001",
                "suite": "image_static",
                "prediction_stem": "sample-001",
                "conditioning_image": str(input_path),
                "output_path": str(output_path),
                "prompt": "Camera pans left across a quiet coast.",
                "camera_path": ["pan_left", "push_in"],
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
            module,
            "--repo_root",
            str(config.repo_root),
            "--checkpoint_dir",
            str(config.checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            env_name: "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cami2v_runner_dry_run_resolves_released_pt_checkpoint(tmp_path: Path) -> None:
    payload = _run_dry_runner(
        tmp_path=tmp_path,
        module="worldarena.models.adapters.cami2v_batch_runner",
        env_name="WORLDARENA_CAMI2V_DRY_RUN",
        config_path="config/models/cami2v.yaml",
    )
    assert payload["dry_run"] is True
    assert str(payload["checkpoint_file"]).endswith("512_cami2v_100k.pt")
    assert payload["sample_preview"]["camera_path"] == ["pan_left", "push_in"]
    assert payload["video_length"] == 16


def test_realcam_i2v_runner_dry_run_uses_local_cogvideox_base(tmp_path: Path) -> None:
    payload = _run_dry_runner(
        tmp_path=tmp_path,
        module="worldarena.models.adapters.realcam_i2v_batch_runner",
        env_name="WORLDARENA_REALCAM_I2V_DRY_RUN",
        config_path="config/models/realcam_i2v.yaml",
    )
    assert payload["dry_run"] is True
    assert str(payload["pretrained_model_path"]).endswith("CogVideoX1.5-5B-I2V")
    assert str(payload["controlnet_checkpoint"]).endswith(
        "RealCam-I2V_CogVideoX1.5-5B_50k.pt"
    )
    assert payload["video_shape"] == [81, 512, 896]
    assert payload["sample_preview"]["camera_path"] == ["pan_left", "push_in"]
