from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_gen3c_batch import _batch_input_payload, _output_signature
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.gen3c import (
    Gen3CAdapter,
    _generation_for_sample,
)
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.gen3c_runner import generate_gen3c_camera_path
from worldarena.models.config import ModelRuntimeConfig, load_model_config


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
        supported_suites=["image_static"],
        prompt_mode="prompt_current",
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


def test_gen3c_image_static_uses_scene_only_prompt_current(tmp_path: Path) -> None:
    sample = _make_image_static_sample(tmp_path / "reference.png", camera_path=["orbit_left"])
    adapter = Gen3CAdapter(_make_config(tmp_path))

    assert adapter.prompt_for(sample) == sample.prompt_current


def test_gen3c_config_only_supports_image_static() -> None:
    config = load_model_config(Path("config/models/gen3c.yaml"))

    assert config.supported_suites == ["image_static"]


@pytest.mark.parametrize(
    "camera_token",
    [
        "orbit_left",
        "orbit_right",
        "push_in",
        "pull_out",
        "move_left",
        "move_right",
        "pan_left",
        "pan_right",
    ],
)
def test_gen3c_image_static_preserves_canonical_camera_path(
    tmp_path: Path,
    camera_token: str,
) -> None:
    sample = _make_image_static_sample(tmp_path / "reference.png", camera_path=[camera_token])

    generation = _generation_for_sample({"trajectory": "left", "camera_rotation": "center_facing"}, sample)

    assert generation["camera_path"] == [camera_token]
    assert generation["camera_path_primary"] == camera_token
    assert generation["camera_path_json"] == f'["{camera_token}"]'
    assert generation["camera_control_source"] == "worldarena_canonical_c2w"


@pytest.mark.parametrize(
    "camera_token",
    [
        "push_in",
        "pull_out",
        "move_left",
        "move_right",
        "pan_left",
        "pan_right",
        "orbit_left",
        "orbit_right",
    ],
)
def test_gen3c_eight_diagram_controls_match_canonical_geometry(camera_token: str) -> None:
    import torch

    from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices

    initial_w2c = torch.eye(4)
    initial_intrinsics = torch.tensor(
        [[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]]
    )
    w2cs, intrinsics = generate_gen3c_camera_path(
        camera_path=[camera_token],
        generate_camera_trajectory=None,
        initial_w2c=initial_w2c,
        initial_intrinsics=initial_intrinsics,
        num_frames=13,
        movement_distance=1.0,
        device="cpu",
    )
    generated_c2w = torch.linalg.inv(w2cs[0])
    expected_c2w = torch.from_numpy(
        synthetic_camera_matrices([camera_token], target_frames=13).matrices
    )

    assert w2cs.shape == (1, 13, 4, 4)
    assert intrinsics.shape == (1, 13, 3, 3)
    assert torch.allclose(generated_c2w, expected_c2w, atol=1e-5)
    assert torch.equal(intrinsics[0, 0], initial_intrinsics)
    assert torch.equal(intrinsics[0, -1], initial_intrinsics)


def test_gen3c_pan_is_rotation_only_and_move_is_translation_only() -> None:
    import torch

    initial_w2c = torch.eye(4)
    initial_intrinsics = torch.eye(3)
    pan_w2cs, _ = generate_gen3c_camera_path(
        camera_path=["pan_left"],
        generate_camera_trajectory=None,
        initial_w2c=initial_w2c,
        initial_intrinsics=initial_intrinsics,
        num_frames=9,
        movement_distance=1.0,
        device="cpu",
    )
    move_w2cs, _ = generate_gen3c_camera_path(
        camera_path=["move_left"],
        generate_camera_trajectory=None,
        initial_w2c=initial_w2c,
        initial_intrinsics=initial_intrinsics,
        num_frames=9,
        movement_distance=1.0,
        device="cpu",
    )
    pan_c2w = torch.linalg.inv(pan_w2cs[0])
    move_c2w = torch.linalg.inv(move_w2cs[0])

    assert torch.allclose(pan_c2w[:, :3, 3], torch.zeros(9, 3), atol=1e-6)
    assert not torch.allclose(pan_c2w[-1, :3, :3], torch.eye(3), atol=1e-6)
    assert torch.allclose(move_c2w[:, :3, :3], torch.eye(3).repeat(9, 1, 1), atol=1e-6)
    assert torch.allclose(move_c2w[-1, :3, 3], torch.tensor([-0.5, 0.0, 0.0]), atol=1e-6)


def test_gen3c_multi_action_path_is_anchored_to_moge_pose_and_scaled() -> None:
    import torch

    from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices

    initial_c2w = torch.eye(4)
    initial_c2w[:3, 3] = torch.tensor([2.0, -1.0, 4.0])
    initial_w2c = torch.linalg.inv(initial_c2w)
    camera_path = ["pan_left", "pull_out", "move_left"]
    w2cs, _ = generate_gen3c_camera_path(
        camera_path=camera_path,
        generate_camera_trajectory=None,
        initial_w2c=initial_w2c,
        initial_intrinsics=torch.eye(3),
        num_frames=10,
        movement_distance=0.3,
        device="cpu",
    )

    relative_c2w = torch.from_numpy(
        synthetic_camera_matrices(camera_path, target_frames=10).matrices
    )
    relative_c2w[:, :3, 3] *= 0.3
    expected_c2w = initial_c2w.unsqueeze(0) @ relative_c2w

    assert torch.allclose(torch.linalg.inv(w2cs[0]), expected_c2w, atol=1e-5)


def test_gen3c_persistent_batch_preserves_full_camera_path(tmp_path: Path) -> None:
    camera_path = ["pan_left", "pull_out", "move_left"]

    payload = _batch_input_payload(
        item={
            "prompt": "Camera pan left, pull out, then move left.",
            "conditioning_image": tmp_path / "reference.png",
            "prediction_path": tmp_path / "prediction.mp4",
            "generation": {
                "camera_path": camera_path,
                "trajectory": "left",
                "camera_rotation": "trajectory_aligned",
                "movement_distance": 0.3,
            },
        },
        error_path=tmp_path / "error.txt",
    )

    assert payload["camera_path"] == camera_path
    assert payload["trajectory"] == "left"
    assert payload["camera_rotation"] == "trajectory_aligned"


def test_gen3c_adapter_invokes_one_resident_worker_for_multiple_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldarena.models.adapters.gen3c as module

    config = _make_config(tmp_path)
    config.generation = {
        "num_gpus": 1,
        "num_video_frames": 9,
        "height": 64,
        "width": 96,
        "fps": 8,
        "seed": 42,
    }
    adapter = Gen3CAdapter(config)
    sample_one = _make_image_static_sample(
        tmp_path / "reference-one.png",
        camera_path=["pan_left", "pull_out"],
    )
    sample_two = deepcopy(sample_one)
    sample_two.sample_id = "image-static-002"
    sample_two.prediction_stem = "image_static_sample_002"
    sample_two.camera_path = ["move_right"]
    output_dir = tmp_path / "predictions"
    requests = [
        PreparedGenerationRequest(
            sample=sample_one,
            conditioning_image=tmp_path / "reference-one.png",
            output_path=output_dir / "one.mp4",
            prompt=sample_one.prompt_current,
        ),
        PreparedGenerationRequest(
            sample=sample_two,
            conditioning_image=tmp_path / "reference-two.png",
            output_path=output_dir / "two.mp4",
            prompt=sample_two.prompt_current,
        ),
    ]

    moge_dir = tmp_path / "moge"
    moge_dir.mkdir()
    (moge_dir / "model.pt").write_bytes(b"moge")
    monkeypatch.setattr(module, "_require_repo_layout", lambda _repo: None)
    monkeypatch.setattr(module, "_checkpoint_layout", lambda _checkpoint: {})

    def prepare_checkpoint_root(work_root: Path, _layout: dict) -> Path:
        checkpoint_root = work_root / "checkpoints"
        checkpoint_root.mkdir()
        return checkpoint_root

    monkeypatch.setattr(module, "_prepare_checkpoint_root", prepare_checkpoint_root)
    monkeypatch.setattr(module, "hf_local_dir", lambda *_args, **_kwargs: moge_dir)
    monkeypatch.setattr(module, "_link_repo_alias", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_build_gen3c_env", lambda *_args, **_kwargs: {})
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, env) -> None:
        del cwd, env
        calls.append(list(command))
        spec_path = Path(command[command.index("--batch_input_path") + 1])
        rows = [json.loads(line) for line in spec_path.read_text().splitlines()]
        assert [row["camera_path"] for row in rows] == [
            ["pan_left", "pull_out"],
            ["move_right"],
        ]
        for row in rows:
            Path(row["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(row["output_path"]).write_bytes(b"video")

    monkeypatch.setattr(module, "run_command", fake_run)

    results = adapter.generate_batch(requests)

    assert len(calls) == 1
    assert all(result["status"] == "generated" for result in results.values())
    assert all(result["checkpoint_load_count"] == 1 for result in results.values())


def test_gen3c_output_signature_rejects_empty_files(tmp_path: Path) -> None:
    output_path = tmp_path / "prediction.mp4"

    assert _output_signature(output_path) is None
    output_path.touch()
    assert _output_signature(output_path) is None
    output_path.write_bytes(b"video")
    assert _output_signature(output_path) is not None


def test_gen3c_empty_prompt_still_fails_through_base_contract(tmp_path: Path) -> None:
    sample = _make_image_static_sample(tmp_path / "reference.png", camera_path=["orbit_left"])
    sample.prompt_target = ""
    sample.prompt_current = ""
    adapter = Gen3CAdapter(_make_config(tmp_path))

    with pytest.raises(ValueError, match="requires non-empty prompt_current"):
        adapter.prompt_for(sample)
