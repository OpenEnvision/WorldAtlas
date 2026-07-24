from __future__ import annotations

from pathlib import Path

import pytest

from scripts.model_batch_preflight import inspect_model
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


@pytest.mark.parametrize(
    ("config_name", "gpus_per_worker", "worker_count"),
    [
        ("abot_world.yaml", 1, 8),
        ("alayaworld.yaml", 1, 8),
        ("lingbot_video_dense.yaml", 1, 8),
        ("lingbot_video_moe.yaml", 8, 1),
        ("lingbot_world.yaml", 8, 1),
        ("lingbot_world_fast.yaml", 8, 1),
        ("lingbot_world_base_act_preview.yaml", 1, 8),
    ],
)
def test_core_models_are_checkpoint_resident_batches(
    config_name: str,
    gpus_per_worker: int,
    worker_count: int,
) -> None:
    payload = inspect_model(
        Path("config/models") / config_name,
        total_gpus=8,
        require_load_once=True,
        validate_paths=False,
    )

    assert payload["supports_batch_generation"] is True
    assert payload["checkpoint_load_policy"] == "load_once"
    assert payload["gpus_per_worker"] == gpus_per_worker
    assert payload["worker_count"] == worker_count


def test_preflight_rejects_legacy_batch_that_reloads_or_is_unverified() -> None:
    with pytest.raises(
        RuntimeError, match="not approved for checkpoint-resident production"
    ):
        inspect_model(
            Path("config/models/matrix_game_35_first.yaml"),
            total_gpus=8,
            require_load_once=True,
            validate_paths=False,
        )


def test_every_active_model_has_an_explicit_checkpoint_batch_policy() -> None:
    config_paths = sorted(Path("config/models").glob("*.yaml"))

    assert config_paths
    for config_path in config_paths:
        config = load_model_config(config_path)
        adapter = build_model_adapter(config)
        policy = adapter.batch_checkpoint_load_policy()
        assert policy in {"load_once", "reload_per_sample"}, config_path.name
        if config.generation.get("require_persistent_batch", False):
            assert policy == "load_once", config_path.name


def test_every_checkpoint_resident_model_has_an_explicit_batch_launcher() -> None:
    scripts = Path("scripts")
    launcher_names = (
        "launch_abot_world_8gpu.sh",
        "launch_alayaworld_8gpu.sh",
        "launch_camera_i2v_8gpu.sh",
        "launch_cosmos_predict25_8gpu.sh",
        "launch_flashworld_8gpu.sh",
        "launch_lingbot_video_8gpu.sh",
        "launch_lingbot_world_8gpu.sh",
        "launch_gen3c_8gpu.sh",
        "launch_hunyuan_gamecraft_8gpu.sh",
        "launch_infinite_world_8gpu.sh",
        "launch_luciddreamer_8gpu.sh",
        "launch_lyra_8gpu.sh",
        "launch_model_batch_8gpu.sh",
        "launch_stable_virtual_camera_8gpu.sh",
        "launch_wonder_8gpu.sh",
        "launch_worldfm_8gpu.sh",
        "launch_yume_8gpu.sh",
    )
    launcher_text = ""
    for name in launcher_names:
        path = scripts / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        launcher_text += text
        assert (
            "launch_model_batch_8gpu.sh" in text
            or name == "launch_model_batch_8gpu.sh"
        )

    for config_path in sorted(Path("config/models").glob("*.yaml")):
        config = load_model_config(config_path)
        adapter = build_model_adapter(config)
        if adapter.batch_checkpoint_load_policy() == "load_once":
            assert config_path.name in launcher_text, config_path.name
