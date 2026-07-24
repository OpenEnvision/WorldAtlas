from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.generation.runner import _select_samples, generate_predictions
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.stable_virtual_camera_batch_runner import (
    WORLD_SCORE_PRIOR_PREFIX,
    _camera_path as _svc_camera_path,
    _camera_token as _svc_camera_token,
    _patch_world_score_camera_prior,
    _traj_prior as _svc_traj_prior,
)
from worldarena.models.config import ModelRuntimeConfig, load_model_config
from worldarena.models.registry import build_model_adapter


class _DummyBatchAdapter(ModelAdapter):
    def __init__(self, config: ModelRuntimeConfig) -> None:
        super().__init__(config)
        self.batch_requests: list[PreparedGenerationRequest] = []

    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, object]]:
        self.batch_requests.extend(requests)
        results: dict[str, dict[str, object]] = {}
        for request in requests:
            request.output_path.write_bytes(b"video")
            results[request.sample.sample_id] = {
                "status": "generated",
                "prediction_path": str(request.output_path),
                "prompt": request.prompt,
            }
        return results

    def generate(self, sample, conditioning_image, output_path, prompt):  # pragma: no cover
        raise AssertionError("batch adapter should not call per-sample generate()")


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "dummy.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="dummy-batch",
        family="dummy",
        python_bin="python",
        repo_root=tmp_path,
        checkpoint_dir=None,
        entrypoint="generate.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_current",
        reference_mode="reference_image",
    )


def _make_family_config(
    tmp_path: Path,
    *,
    family: str,
    backend_type: str = "local_repo",
) -> ModelRuntimeConfig:
    config = _make_config(tmp_path)
    config.name = family
    config.family = family
    config.backend_type = backend_type
    config.repo_root = tmp_path if backend_type != "http_api" else None
    config.checkpoint_dir = tmp_path
    config.api = {"submit_route": "/submit"} if backend_type == "http_api" else {}
    return config


def _make_sample(reference_path: Path, index: int = 1) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"image-static-{index:03d}",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=str(reference_path),
        relative_path=f"static/photorealistic/indoor/dining_spaces/reference_{index:03d}.png",
        reference_path=str(reference_path),
        prediction_stem=f"image_static_sample_{index:03d}",
        conditioning_strategy="reference_image",
        prompt_current="A dining room.",
        prompt_target="Camera move left. A dining room remains still.",
        style="photorealistic",
        environment="indoor",
        scene="dining_spaces",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


def test_generate_predictions_uses_batch_adapter(monkeypatch, tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), color="white").save(reference_path)

    model_config = _make_config(tmp_path)
    sample = _make_sample(reference_path)
    adapter = _DummyBatchAdapter(model_config)
    monkeypatch.setattr("worldarena.generation.runner.build_model_adapter", lambda config: adapter)

    result = generate_predictions(
        model_config=model_config,
        manifest=[sample],
        output_dir=tmp_path / "predictions",
        overwrite=True,
    )

    assert result["manifest"]["generated"] == 1
    assert len(adapter.batch_requests) == 1
    request = adapter.batch_requests[0]
    assert request.sample.sample_id == sample.sample_id
    assert request.output_path.exists()
    assert result["records"][0]["status"] == "generated"


def test_registered_model_adapters_expose_batch_generation(tmp_path: Path) -> None:
    families = [
        "lingbot_world",
        "gen3c",
        "luciddreamer",
        "hy_worldplay",
        "hunyuan_gamecraft",
        "hunyuanworld",
        "matrix_game",
        "lyra1",
        "lyra2",
        "flashworld",
        "inspatio_world",
        "infinite_world",
        "neoverse",
        "stable_virtual_camera",
        "wonderworld",
        "wonderjourney",
        "worldcam",
        "worldfm",
        "yume",
    ]
    adapters = [
        build_model_adapter(_make_family_config(tmp_path, family=family))
        for family in families
    ]
    adapters.append(
        build_model_adapter(
            _make_family_config(tmp_path, family="video_api", backend_type="http_api")
        )
    )

    for adapter in adapters:
        assert adapter.supports_batch_generation()
        assert adapter.generate_batch.__func__ is not ModelAdapter.generate_batch


def test_every_active_model_config_builds_a_registered_adapter() -> None:
    config_root = Path(__file__).resolve().parents[1] / "config" / "models"
    config_paths = sorted(config_root.glob("*.yaml"))

    assert config_paths
    for config_path in config_paths:
        config = load_model_config(config_path)
        adapter = build_model_adapter(config)
        assert adapter.config is config, config_path.name


def test_generate_predictions_shards_batch_outputs(monkeypatch, tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), color="white").save(reference_path)

    model_config = _make_config(tmp_path)
    manifest = [_make_sample(reference_path, index=index) for index in range(4)]
    adapter = _DummyBatchAdapter(model_config)
    monkeypatch.setattr("worldarena.generation.runner.build_model_adapter", lambda config: adapter)

    output_dir = tmp_path / "predictions"
    result = generate_predictions(
        model_config=model_config,
        manifest=manifest,
        output_dir=output_dir,
        overwrite=True,
        num_shards=2,
        shard_index=1,
    )

    assert result["manifest"]["num_shards"] == 2
    assert result["manifest"]["shard_index"] == 1
    assert [request.sample.sample_id for request in adapter.batch_requests] == [
        manifest[1].sample_id,
        manifest[3].sample_id,
    ]
    assert (output_dir / "generation_manifest.shard1.json").exists()
    assert (output_dir / "generation_records.shard1.jsonl").exists()
    assert (output_dir / "prompt_map.shard1.json").exists()
    assert not (output_dir / "generation_manifest.json").exists()


def test_select_samples_limit_zero_returns_empty(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), color="white").save(reference_path)

    selected = _select_samples(
        [_make_sample(reference_path)],
        suites=None,
        limit=0,
        supported_suites={"image_static"},
    )

    assert selected == []


def test_stable_virtual_camera_config_excludes_dynamic_i2v_suite() -> None:
    config = load_model_config(
        Path(__file__).resolve().parents[1] / "config" / "models" / "stable_virtual_camera.yaml"
    )

    assert config.supported_suites == ["image_static"]


def test_stable_virtual_camera_preserves_complete_camera_path() -> None:
    assert _svc_camera_token({"camera_path": ["fixed"]}) == "fixed"
    assert _svc_camera_token({}) == "fixed"
    assert _svc_camera_path({"camera_path": ["pan-left", "move right", "pull_out"]}) == [
        "pan_left",
        "move_right",
        "pull_out",
    ]
    assert _svc_traj_prior({"camera_path": ["fixed"]}, {}) == (
        WORLD_SCORE_PRIOR_PREFIX + "fixed"
    )
    assert _svc_traj_prior({"camera_path": ["pan_left", "pull_out"]}, {}) == (
        WORLD_SCORE_PRIOR_PREFIX + "pan_left,pull_out"
    )


def test_stable_virtual_camera_rejects_unknown_camera_token() -> None:
    with pytest.raises(ValueError, match="unsupported Stable Virtual Camera token"):
        _svc_traj_prior({"camera_path": ["teleport"]}, {})


def test_stable_virtual_camera_prior_matches_world_score_trajectory() -> None:
    import numpy as np
    import torch

    from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices

    calls: list[tuple[str, dict[str, object]]] = []

    def get_preset_pose_fov(option, num_frames, start_w2c, look_at, *args, **kwargs):
        del num_frames, start_w2c, look_at, args
        calls.append((option, kwargs))
        return np.zeros((7, 4, 4), dtype=np.float32), np.full((7,), 0.8)

    module = SimpleNamespace(get_preset_pose_fov=get_preset_pose_fov, torch=torch)
    _patch_world_score_camera_prior(module)

    camera_path = ["pan_left", "move_right", "pull_out"]
    poses, fovs = module.get_preset_pose_fov(
        WORLD_SCORE_PRIOR_PREFIX + ",".join(camera_path),
        7,
        torch.eye(4),
        torch.tensor([0.0, 0.0, 10.0]),
    )

    expected = synthetic_camera_matrices(camera_path, target_frames=7).matrices
    np.testing.assert_allclose(poses, expected, atol=1e-6)
    np.testing.assert_allclose(fovs, np.full((7,), 0.8))
    assert calls == [("zoom-in", {"zoom_factor": 1.0})]


def test_stable_virtual_camera_fixed_prior_keeps_pose_and_fov_constant() -> None:
    import numpy as np
    import torch

    def get_preset_pose_fov(option, num_frames, start_w2c, look_at, *args, **kwargs):
        del option, start_w2c, look_at, args, kwargs
        return np.zeros((num_frames, 4, 4)), np.full((num_frames,), 0.8)

    module = SimpleNamespace(get_preset_pose_fov=get_preset_pose_fov, torch=torch)
    _patch_world_score_camera_prior(module)
    poses, fovs = module.get_preset_pose_fov(
        WORLD_SCORE_PRIOR_PREFIX + "fixed",
        8,
        torch.eye(4),
        torch.tensor([0.0, 0.0, 10.0]),
    )

    np.testing.assert_allclose(poses, np.repeat(np.eye(4)[None], 8, axis=0), atol=1e-6)
    np.testing.assert_allclose(fovs, np.full((8,), 0.8))


def test_stable_virtual_camera_left_right_actions_have_opposite_world_score_directions() -> None:
    import numpy as np
    import torch

    def get_preset_pose_fov(option, num_frames, start_w2c, look_at, *args, **kwargs):
        del option, start_w2c, look_at, args, kwargs
        return np.zeros((num_frames, 4, 4)), np.full((num_frames,), 0.8)

    module = SimpleNamespace(get_preset_pose_fov=get_preset_pose_fov, torch=torch)
    _patch_world_score_camera_prior(module)

    def trajectory(token: str) -> np.ndarray:
        poses, _ = module.get_preset_pose_fov(
            WORLD_SCORE_PRIOR_PREFIX + token,
            8,
            torch.eye(4),
            torch.tensor([0.0, 0.0, 10.0]),
        )
        return poses

    assert trajectory("move_left")[-1, 0, 3] < 0
    assert trajectory("move_right")[-1, 0, 3] > 0
    assert trajectory("pan_left")[-1, 0, 2] < 0
    assert trajectory("pan_right")[-1, 0, 2] > 0
    assert trajectory("orbit_left")[-1, 0, 3] < 0
    assert trajectory("orbit_right")[-1, 0, 3] > 0
