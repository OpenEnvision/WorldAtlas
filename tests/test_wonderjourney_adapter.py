from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
import torch

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common import checkpoints
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.wonderjourney import WonderJourneyAdapter
from worldarena.models.adapters.wonderjourney_runner import (
    WONDERJOURNEY_MOTION_CODES,
    WONDERJOURNEY_MOTION_DEFINITIONS,
    _runtime_entrypoint,
    _validated_motion_plan,
    install_wonderjourney_camera_control,
    parse_args,
    resolve_wonderjourney_camera_control,
)
from worldarena.models.adapters.wonderworld import _generation_for_suite
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _patch_workspace(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "WorldAtlas Arena"
    ckpt_root = workspace_root / "ckpt"
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", workspace_root)
    monkeypatch.setattr(checkpoints, "CANONICAL_CHECKPOINT_ROOT", ckpt_root)
    return workspace_root, ckpt_root


def _make_config(tmp_path: Path, monkeypatch) -> ModelRuntimeConfig:
    workspace_root, ckpt_root = _patch_workspace(monkeypatch, tmp_path)
    config_path = workspace_root / "config" / "models" / "wonderjourney.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")

    repo_root = workspace_root / "thirdparty" / "WonderJourney"
    (repo_root / "config").mkdir(parents=True, exist_ok=True)
    (repo_root / "run.py").write_text("print('stub')\n", encoding="utf-8")
    (repo_root / "config" / "base-config.yaml").write_text("device: cuda\n", encoding="utf-8")

    midas_checkpoint = ckpt_root / "WonderJourney" / "dpt_beit_large_512.pt"
    midas_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    midas_checkpoint.write_bytes(b"ckpt")

    return ModelRuntimeConfig(
        config_path=config_path,
        name="wonderjourney",
        family="wonderjourney",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=None,
        entrypoint="run.py",
        output_ext=".mp4",
        supported_suites=["image_static", "video_static"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        control_signals=["text", "image"],
        generation={
            "midas_checkpoint": str(midas_checkpoint),
            "stable_diffusion_checkpoint": "ckpt/stabilityai--stable-diffusion-2-inpainting",
            "entity_limit": 2,
            "frames": 4,
            "num_scenes": 1,
            "num_keyframes": 2,
            "rotation_path": [0, 1],
            "use_sample_camera_path": True,
            "rotation_range": 0.45,
            "camera_speed_multiplier_rotation": 0.2,
            "save_fps": 8,
            "finetune_decoder_gen": False,
            "finetune_depth_model": False,
            "archive_sidecars": True,
            "frames_by_suite": {"video_static": 6},
        },
    )


def _make_sample(
    suite: str = "video_static",
    *,
    camera_path: list[str] | None = None,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level="video" if suite == "video_static" else "image",
        modality="video" if suite == "video_static" else "image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=f"/dataset/{suite}.mp4" if suite == "video_static" else f"/dataset/{suite}.png",
        relative_path=f"{suite}/sample",
        reference_path=f"/dataset/{suite}.mp4" if suite == "video_static" else f"/dataset/{suite}.png",
        prediction_stem=f"{suite}_sample",
        conditioning_strategy="first_frame",
        prompt_current="A static dining room scene.",
        prompt_target="Camera fixed. A static dining room scene.",
        style="photorealistic",
        environment="indoor",
        scene="dining space",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=list(camera_path or []),
    )


def test_generation_for_suite_applies_by_suite_overrides() -> None:
    resolved = _generation_for_suite(
        {
            "frames": 4,
            "frames_by_suite": {"video_static": 6},
        },
        "video_static",
    )

    assert resolved["frames"] == 6


@pytest.mark.parametrize(
    ("action", "native_rotation", "translation", "pan"),
    [
        ("fixed", 0, (0.0, 0.0, 0.0), False),
        ("push_in", 0, (0.0, 0.0, 1.0), False),
        ("pull_out", 0, (0.0, 0.0, -1.0), False),
        ("move_left", 0, (1.0, 0.0, 0.0), False),
        ("move_right", 0, (-1.0, 0.0, 0.0), False),
        ("pan_left", -1, None, True),
        ("pan_right", 1, None, True),
        ("orbit_left", -1, None, False),
        ("orbit_right", 1, None, False),
    ],
)
def test_camera_action_definitions_are_explicit(
    action: str,
    native_rotation: int,
    translation: tuple[float, float, float] | None,
    pan: bool,
) -> None:
    definition = WONDERJOURNEY_MOTION_DEFINITIONS[WONDERJOURNEY_MOTION_CODES[action]]

    assert definition["action"] == action
    assert definition["native_rotation"] == native_rotation
    assert definition["translation"] == translation
    assert bool(definition.get("pan", False)) is pan


def test_camera_control_repeats_single_action_across_configured_segments() -> None:
    control = resolve_wonderjourney_camera_control(
        ["pan_right"],
        frames=50,
        num_scenes=1,
        num_keyframes=2,
    )

    assert control["camera_actions"] == ["pan_right", "pan_right"]
    assert control["rotation_path"] == [
        WONDERJOURNEY_MOTION_CODES["pan_right"],
        WONDERJOURNEY_MOTION_CODES["pan_right"],
    ]
    assert control["frames"] == 50
    assert control["num_keyframes"] == 2


def test_camera_control_preserves_three_action_sequence_and_frame_budget() -> None:
    actions = ["pan_left", "pull_out", "move_left"]

    control = resolve_wonderjourney_camera_control(
        actions,
        frames=50,
        num_scenes=1,
        num_keyframes=2,
    )

    assert control["camera_actions"] == actions
    assert control["rotation_path"] == [WONDERJOURNEY_MOTION_CODES[action] for action in actions]
    assert control["frames"] == 33
    assert control["num_keyframes"] == 3
    assert control["total_frame_budget"] == 100


def test_camera_control_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="does not support"):
        resolve_wonderjourney_camera_control(
            ["teleport"],
            frames=50,
            num_scenes=1,
            num_keyframes=2,
        )


def test_motion_plan_rejects_pan_with_zero_rotation_range() -> None:
    args = parse_args(
        [
            "--rotation_path",
            json.dumps([WONDERJOURNEY_MOTION_CODES["pan_right"]] * 2),
            "--camera_actions",
            json.dumps(["pan_right", "pan_right"]),
            "--num_scenes",
            "1",
            "--num_keyframes",
            "2",
            "--rotation_range",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="non-zero rotation_range"):
        _validated_motion_plan(args)


def test_registry_builds_wonderjourney_adapter(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, WonderJourneyAdapter)


def test_runtime_entrypoint_removes_default_fp16_revision(tmp_path: Path) -> None:
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text(
        'pipe = StableDiffusionInpaintPipeline.from_pretrained("repo", revision="fp16",)\n',
        encoding="utf-8",
    )

    patched = _runtime_entrypoint(entrypoint, tmp_path, None)

    assert patched != entrypoint
    assert 'revision="fp16"' not in patched.read_text(encoding="utf-8")


def test_runtime_entrypoint_installs_camera_control_patch(tmp_path: Path) -> None:
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text(
        "from models.models import KeyframeGen, KeyframeInterp, save_point_cloud_as_ply\n"
        "import sys\n"
        "if __name__ == '__main__':\n"
        "    config = OmegaConf.merge(base_config, example_config)\n"
        "    run(config)\n",
        encoding="utf-8",
    )

    patched = _runtime_entrypoint(
        entrypoint,
        tmp_path,
        None,
        require_camera_control=True,
    )
    source = patched.read_text(encoding="utf-8")

    assert "install_wonderjourney_camera_control" in source
    assert "install_wonderjourney_camera_control(sys.modules[__name__])" in source


def test_runtime_entrypoint_rejects_incompatible_upstream_when_control_required(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text("print('incompatible')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incompatible"):
        _runtime_entrypoint(
            entrypoint,
            tmp_path,
            None,
            require_camera_control=True,
        )


def test_runtime_camera_patch_applies_linear_motion_and_native_pan() -> None:
    module_name = "test_fake_wonderjourney_models"

    class FakeFrameSyn:
        def get_next_camera_rotation(self):
            return "native-motion"

    class FakeKeyframeGen(FakeFrameSyn):
        def __init__(self, config, inpainter, mask, depth, vae, rotation, *args, **kwargs):
            del config, inpainter, mask, depth, vae, args, kwargs
            self.received_rotation = rotation
            self.current_camera = SimpleNamespace(
                T=torch.zeros((1, 3), dtype=torch.float32),
                rotations_count=0,
            )
            self.camera_speed = 0.5

    class FakeKeyframeInterp(FakeFrameSyn):
        def __init__(self, config, inpainter, depth, vae, rotation, *args, **kwargs):
            del config, inpainter, depth, vae, args, kwargs
            self.received_rotation = rotation
            self.current_camera = SimpleNamespace(
                T=torch.zeros((1, 3), dtype=torch.float32),
                rotations_count=1,
            )
            self.camera_speed = 0.5

    FakeFrameSyn.__module__ = module_name
    FakeKeyframeGen.__module__ = module_name
    FakeKeyframeInterp.__module__ = module_name
    models_module = ModuleType(module_name)
    models_module.FrameSyn = FakeFrameSyn
    models_module.KeyframeGen = FakeKeyframeGen
    models_module.KeyframeInterp = FakeKeyframeInterp
    sys.modules[module_name] = models_module
    runtime = SimpleNamespace(KeyframeGen=FakeKeyframeGen, KeyframeInterp=FakeKeyframeInterp)
    try:
        install_wonderjourney_camera_control(runtime)

        pull_out = runtime.KeyframeGen(
            None,
            None,
            None,
            None,
            None,
            WONDERJOURNEY_MOTION_CODES["pull_out"],
        )
        pull_camera = pull_out.get_next_camera_rotation()
        assert pull_out.received_rotation == 0
        torch.testing.assert_close(
            pull_camera.T,
            torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float32),
        )

        move_right = runtime.KeyframeInterp(
            None,
            None,
            None,
            None,
            WONDERJOURNEY_MOTION_CODES["move_right"],
        )
        right_camera = move_right.get_next_camera_rotation()
        assert move_right.received_rotation == 0
        assert right_camera.rotations_count == 2
        torch.testing.assert_close(
            right_camera.T,
            torch.tensor([[-0.5, 0.0, 0.0]], dtype=torch.float32),
        )

        pan_right = runtime.KeyframeGen(
            None,
            None,
            None,
            None,
            None,
            WONDERJOURNEY_MOTION_CODES["pan_right"],
        )
        assert pan_right.received_rotation == 1
        assert pan_right.camera_speed == 0.0
        assert pan_right.get_next_camera_rotation() == "native-motion"
    finally:
        sys.modules.pop(module_name, None)


def test_generate_builds_runner_command(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, monkeypatch)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample()
    recorded: dict[str, object] = {}

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        output_path.write_bytes(b"mp4")
        output_path.with_name(f"{output_path.stem}.reverse{output_path.suffix}").write_bytes(
            b"reverse"
        )
        output_path.with_name(f"{output_path.stem}.keyframes{output_path.suffix}").write_bytes(
            b"keyframes"
        )

    monkeypatch.setattr("worldarena.models.adapters.wonderjourney.run_command", fake_run_command)

    adapter = WonderJourneyAdapter(config)
    result = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    command = recorded["command"]
    assert isinstance(command, list)
    assert "worldarena.models.adapters.wonderjourney_runner" in command
    assert "--midas_checkpoint" in command
    assert "--stable_diffusion_checkpoint" in command
    assert "--rotation_path" in command
    assert "--frames" in command
    assert "6" in command
    assert result["prediction_path"] == str(output_path)
    assert result["scene_name"] == "dining space"
    assert result["reverse_video_path"] == str(
        output_path.with_name(f"{output_path.stem}.reverse{output_path.suffix}")
    )
    assert result["keyframes_video_path"] == str(
        output_path.with_name(f"{output_path.stem}.keyframes{output_path.suffix}")
    )


def test_generate_uses_sample_camera_path_instead_of_global_forward_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, monkeypatch)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample("image_static", camera_path=["pan_right"])
    recorded: dict[str, object] = {}

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        recorded["command"] = command
        output_path.write_bytes(b"mp4")

    monkeypatch.setattr("worldarena.models.adapters.wonderjourney.run_command", fake_run_command)

    result = WonderJourneyAdapter(config).generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    command = recorded["command"]
    assert isinstance(command, list)
    rotation_path = json.loads(command[command.index("--rotation_path") + 1])
    camera_actions = json.loads(command[command.index("--camera_actions") + 1])
    assert rotation_path == [
        WONDERJOURNEY_MOTION_CODES["pan_right"],
        WONDERJOURNEY_MOTION_CODES["pan_right"],
    ]
    assert camera_actions == ["pan_right", "pan_right"]
    assert result["camera_path"] == ["pan_right"]
    assert result["camera_actions"] == ["pan_right", "pan_right"]
    assert result["camera_control_source"] == "sample.camera_path"


def test_generate_batch_preserves_full_multi_action_camera_path(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, monkeypatch)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "prediction.mp4"
    actions = ["pan_left", "pull_out", "move_left"]
    sample = _make_sample("image_static", camera_path=actions)

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        spec_path = Path(command[command.index("--batch_spec") + 1])
        results_path = Path(command[command.index("--batch_results") + 1])
        row = json.loads(spec_path.read_text(encoding="utf-8").strip())
        args = row["args"]
        assert json.loads(args[args.index("--camera_actions") + 1]) == actions
        assert json.loads(args[args.index("--rotation_path") + 1]) == [
            WONDERJOURNEY_MOTION_CODES[action] for action in actions
        ]
        assert args[args.index("--num_keyframes") + 1] == "3"
        assert args[args.index("--frames") + 1] == "3"
        output_path.write_bytes(b"mp4")
        results_path.write_text(
            json.dumps(
                {
                    sample.sample_id: {
                        "status": "generated",
                        "prediction_path": str(output_path),
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("worldarena.models.adapters.wonderjourney.run_command", fake_run_command)
    request = PreparedGenerationRequest(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    result = WonderJourneyAdapter(config).generate_batch([request])[sample.sample_id]

    assert result["camera_path"] == actions
    assert result["camera_actions"] == actions
    assert result["wonderjourney_motion_codes"] == [
        WONDERJOURNEY_MOTION_CODES[action] for action in actions
    ]
    assert result["camera_control_frames_per_segment"] == 3
