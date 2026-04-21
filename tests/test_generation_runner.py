from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from worldarena.generation import generate_predictions
from worldarena.models.adapters.matrix_game import _build_action_payload
from worldarena.models.adapters.worldcam_runner import _existing_base_model_root
from worldarena.models.config import ModelRuntimeConfig
from worldarena.benchmark.schemas import BenchmarkSample


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (32, 24))
    assert writer.isOpened()
    for color in ((10, 20, 30), (40, 50, 60), (70, 80, 90)):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :] = color
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _sample(video_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video_static_photorealistic_living_spaces_0001",
        suite="video_static",
        split="dev_public",
        asset_level="video_static",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/living_spaces/0-5s",
        path=str(video_path),
        relative_path="static/photorealistic/indoor/living_spaces/0-5s/0001.avi",
        reference_path=str(video_path),
        prediction_stem="video_static_photorealistic_living_spaces_0001",
        conditioning_strategy="first_frame",
        prompt_current="A photorealistic indoor living spaces scene.",
        prompt_target="Generate a photorealistic video of a coherent indoor living spaces scene.",
        style="photorealistic",
        environment="indoor",
        scene="living_spaces",
        motion_category=None,
        source_name="worldarena_formal_video",
        source_type="curated_video",
        source_group_id="static/photorealistic/indoor/living_spaces::0001",
        license_bucket="C",
        has_annotation=False,
        has_instruction=False,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=32,
        height=24,
        fps=6.0,
        duration_seconds=0.5,
        duration_bucket="0-5s",
    )


def test_generation_runner_extracts_conditioning_frame_and_writes_records(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake_lingbot"
    fake_repo.mkdir(parents=True, exist_ok=True)
    (fake_repo / "generate.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--save_file", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--ckpt_dir")
parser.add_argument("--task")
parser.add_argument("--base_seed")
parser.add_argument("--size")
parser.add_argument("--frame_num")
parser.add_argument("--sample_solver")
parser.add_argument("--sample_steps")
parser.add_argument("--sample_shift")
parser.add_argument("--sample_guide_scale")
parser.add_argument("--ulysses_size")
parser.add_argument("--prompt_extend_method")
parser.add_argument("--prompt_extend_model")
parser.add_argument("--prompt_extend_target_lang")
parser.add_argument("--offload_model")
parser.add_argument("--use_prompt_extend", action="store_true")
parser.add_argument("--t5_fsdp", action="store_true")
parser.add_argument("--t5_cpu", action="store_true")
parser.add_argument("--dit_fsdp", action="store_true")
parser.add_argument("--convert_model_dtype", action="store_true")
args = parser.parse_args()
output = Path(args.save_file)
output.write_bytes(b"fake-video")
Path(str(output) + ".meta.json").write_text(
    json.dumps(
        {
            "image": args.image,
            "prompt": args.prompt,
            "sample_steps": args.sample_steps,
            "sample_shift": args.sample_shift,
            "sample_guide_scale": args.sample_guide_scale,
            "ulysses_size": args.ulysses_size,
            "prompt_extend_method": args.prompt_extend_method,
            "prompt_extend_model": args.prompt_extend_model,
            "prompt_extend_target_lang": args.prompt_extend_target_lang,
            "offload_model": args.offload_model,
            "use_prompt_extend": args.use_prompt_extend,
            "t5_fsdp": args.t5_fsdp,
            "t5_cpu": args.t5_cpu,
            "dit_fsdp": args.dit_fsdp,
            "convert_model_dtype": args.convert_model_dtype,
        }
    ),
    encoding="utf-8",
)
""".strip(),
        encoding="utf-8",
    )

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "lingbot.yaml",
        name="lingbot-world",
        family="lingbot_world",
        python_bin=sys.executable,
        repo_root=fake_repo,
        checkpoint_dir=tmp_path / "checkpoints",
        entrypoint="generate.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        generation={
            "task": "i2v-A14B",
            "frame_num": 9,
            "size": "320*240",
            "base_seed": 7,
            "sample_steps": 4,
            "sample_shift": 5.5,
            "sample_guide_scale": 6.0,
            "ulysses_size": 2,
            "prompt_extend_method": "local_qwen",
            "prompt_extend_model": "qwen-smoke",
            "prompt_extend_target_lang": "en",
            "offload_model": True,
            "use_prompt_extend": True,
            "t5_fsdp": True,
            "t5_cpu": True,
            "dit_fsdp": True,
            "convert_model_dtype": True,
        },
    )

    result = generate_predictions(
        model_config=model_config,
        manifest=[_sample(video_path)],
        output_dir=tmp_path / "predictions",
        suites=None,
        splits=None,
        limit=None,
        overwrite=True,
    )

    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
    assert output_path.exists()
    meta = json.loads((Path(str(output_path) + ".meta.json")).read_text(encoding="utf-8"))
    assert meta["prompt"] == "Generate a photorealistic video of a coherent indoor living spaces scene."
    assert Path(meta["image"]).exists()
    assert Path(meta["image"]).name == "video_static_photorealistic_living_spaces_0001.png"
    assert meta["sample_steps"] == "4"
    assert meta["sample_shift"] == "5.5"
    assert meta["sample_guide_scale"] == "6.0"
    assert meta["ulysses_size"] == "2"
    assert meta["prompt_extend_method"] == "local_qwen"
    assert meta["prompt_extend_model"] == "qwen-smoke"
    assert meta["prompt_extend_target_lang"] == "en"
    assert meta["offload_model"] == "true"
    assert meta["use_prompt_extend"] is True
    assert meta["t5_fsdp"] is True
    assert meta["t5_cpu"] is True
    assert meta["dit_fsdp"] is True
    assert meta["convert_model_dtype"] is True
    manifest = json.loads((tmp_path / "predictions" / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated"] == 1
    assert result["records"][0]["status"] == "generated"


def test_matrix_game_action_payload_uses_suite_specific_plan() -> None:
    payload = _build_action_payload(
        {
            "action_plan": {
                "default": ["forward"],
                "by_suite": {"video_dynamic": ["forward", "camera_r", "forward_right"]},
                "frames_per_action": 4,
                "first_clip_frames": 12,
                "clip_overlap_frames": 8,
                "num_iterations": 2,
            }
        },
        "video_dynamic",
    )
    assert payload["actions"] == ["forward", "camera_r", "forward_right"]
    assert payload["num_iterations"] == 2
    assert len(payload["keyboard_condition"]) == 60
    assert len(payload["mouse_condition"]) == 60


def test_worldcam_adapter_uses_runner_and_annotation_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr("worldarena.models.adapters.worldcam.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        annotation_dir / "poses.npy",
        np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0),
    )
    np.save(
        annotation_dir / "intrinsics.npy",
        np.asarray([[20.0, 20.0, 8.0, 6.0]] * 2, dtype=np.float32),
    )

    sample = _sample(video_path)
    sample = BenchmarkSample(
        **{
            **sample.to_dict(),
            "annotation_path": str(annotation_dir),
            "has_pose": True,
        }
    )
    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "worldcam.yaml",
        name="worldcam",
        family="worldcam",
        python_bin=sys.executable,
        repo_root=tmp_path / "WorldCam",
        checkpoint_dir=tmp_path / "weights",
        entrypoint="inference.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        generation={
            "prompt_prefix": "<fps> ",
            "camera_frames": 233,
            "weights_filename": "finetuned_dit.safetensors",
        },
    )

    result = generate_predictions(
        model_config=model_config,
        manifest=[sample],
        output_dir=tmp_path / "predictions",
        suites=None,
        splits=None,
        limit=None,
        overwrite=True,
    )

    assert len(commands) == 1
    command, cwd, env = commands[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        "worldarena.models.adapters.worldcam_runner",
    ]
    assert command[command.index("--input_video") + 1] == str(video_path.resolve())
    assert command[command.index("--annotation_path") + 1] == str(annotation_dir.resolve())
    assert command[command.index("--checkpoint_dir") + 1] == str((tmp_path / "weights").resolve())
    assert command[command.index("--prompt") + 1].startswith("<fps> A photorealistic indoor")
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
    assert output_path.exists()
    assert result["records"][0]["status"] == "generated"


def test_worldcam_runner_reuses_sibling_wan_cache(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "cache" / "hfd"
    checkpoint_dir = cache_root / "worldcam--worldcam"
    wan_cache = cache_root / "Wan-AI--Wan2.1-T2V-1.3B"
    wan_cache.mkdir(parents=True, exist_ok=True)
    for filename in (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
    ):
        (wan_cache / filename).write_bytes(b"x")

    monkeypatch.setenv("HF_HOME", str(cache_root))

    resolved = _existing_base_model_root(checkpoint_dir)

    assert resolved == wan_cache.resolve()
