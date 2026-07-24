from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.matrix_game_35 import MatrixGame35Adapter
from worldarena.models.adapters.matrix_game_35_batch_runner import (
    _upstream_environment,
    _write_action_sidecar,
    write_camera_npz,
)
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def _sample(*, suite: str = "image_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="matrix-game-35-sample",
        suite=suite,
        split="test",
        asset_level=suite,
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/outdoor/test_scene/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem="matrix_game_35_sample",
        conditioning_strategy="reference_image",
        prompt_current="A quiet outdoor scene.",
        prompt_target="Camera orbits left. A quiet outdoor scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=["orbit_left", "push_in"],
        generation_mode="dynamic" if suite.endswith("_dynamic") else "static",
    )


def _request(tmp_path: Path, *, suite: str = "image_static") -> PreparedGenerationRequest:
    return PreparedGenerationRequest(
        sample=_sample(suite=suite),
        conditioning_image=tmp_path / "input.png",
        output_path=tmp_path / "output.mp4",
        prompt="Camera orbits left. A quiet outdoor scene.",
    )


def _dummy_checkpoint_layout(root: Path) -> None:
    files = (
        "Matrix-Game-3.5-Base/first-person.safetensors",
        "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00001.safetensors",
        "Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer_config.json",
        "DA3NESTED-GIANT-LARGE-1.1/config.json",
        "DA3NESTED-GIANT-LARGE-1.1/model.safetensors",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")


def test_matrix_game_35_configs_select_the_new_adapter() -> None:
    for config_name, person in (
        ("matrix_game_35_first.yaml", "first"),
        ("matrix_game_35_third.yaml", "third"),
    ):
        config = load_model_config(Path("config/models") / config_name)
        adapter = build_model_adapter(config)
        assert isinstance(adapter, MatrixGame35Adapter)
        assert config.generation["person"] == person


def test_matrix_game_35_payload_uses_fixed_camera_for_dynamic_samples(tmp_path: Path) -> None:
    config = load_model_config(Path("config/models/matrix_game_35_first.yaml"))
    adapter = MatrixGame35Adapter(config)

    static_payload = adapter.batch_spec_payload(_request(tmp_path))
    dynamic_payload = adapter.batch_spec_payload(_request(tmp_path, suite="image_dynamic"))

    assert static_payload["camera_path"] == ["orbit_left", "push_in"]
    assert static_payload["camera_source"] == "worldarena_camera_path"
    assert dynamic_payload["camera_path"] == ["fixed"]
    assert dynamic_payload["camera_source"] == "dynamic_fixed"


def test_matrix_game_35_camera_npz_matches_upstream_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (1280, 720), color=(20, 30, 40)).save(image_path)
    destination = tmp_path / "camera.npz"

    metadata = write_camera_npz(
        {
            "conditioning_image": str(image_path),
            "camera_path": ["orbit_left", "push_in"],
        },
        {"num_blocks": 1, "horizontal_fov_degrees": 75.0},
        destination,
    )
    camera = np.load(destination)

    assert camera["extrinsics_c2w"].shape == (86, 4, 4)
    assert camera["intrinsics"].shape == (86, 4)
    assert np.isfinite(camera["extrinsics_c2w"]).all()
    assert camera["intrinsics"][0, 2:].tolist() == [639.5, 359.5]
    assert metadata["camera_path"] == ["orbit_left", "push_in"]
    assert metadata["camera_pose_count"] == 86


def test_matrix_game_35_exposes_vendored_da3_before_upstream_imports(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "Matrix-Game-3.5"
    expected_da3 = repo_root / "third_party" / "depth-anything-3" / "src"

    env = _upstream_environment(repo_root)

    python_paths = env["PYTHONPATH"].split(os.pathsep)
    assert python_paths[:2] == [str(expected_da3), str(repo_root)]


def test_matrix_game_35_sidecar_matches_upstream_output_frame_count(tmp_path: Path) -> None:
    output_path = tmp_path / "prediction.mp4"
    _write_action_sidecar(
        output_path,
        {"camera_path": ["push_in"], "camera_pose_count": 86},
        {"num_blocks": 1},
    )

    payload = json.loads((tmp_path / "prediction_actions.json").read_text())
    assert payload["generated_frames"] == 80
    assert payload["total_frames"] == 85
    assert payload["output_fps"] == 16


def test_matrix_game_35_batch_runner_dry_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "Matrix-Game-3.5"
    repo_root.mkdir()
    (repo_root / "infer.py").write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    _dummy_checkpoint_layout(checkpoint_dir)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (640, 360), color=(20, 30, 40)).save(image_path)

    spec_path = tmp_path / "batch.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "conditioning_image": str(image_path),
                "output_path": str(tmp_path / "output.mp4"),
                "prompt": "Camera moves forward.",
                "camera_path": ["push_in"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.json"
    generation_path.write_text('{"person": "first", "num_blocks": 1}\n', encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldarena.models.adapters.matrix_game_35_batch_runner",
            "--repo_root",
            str(repo_root),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=project_root,
        env={
            **os.environ,
            "PYTHONPATH": str(project_root),
            "WORLDARENA_MATRIX_GAME_35_DRY_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["status"] == "dry_run"
    assert payload["person"] == "first"
    assert payload["camera_path"] == ["push_in"]
    assert Path(payload["camera_npz"]).is_file()
