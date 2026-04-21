from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from worldarena.generation import generate_predictions
from worldarena.models.adapters.hunyuanworld import (
    _label_groups_for_sample as _hunyuanworld_label_groups_for_sample,
)
from worldarena.models.adapters.infinite_world import _build_action_payload as _build_infinite_world_action_payload
from worldarena.models.adapters.lyra1_runner import _checkpoint_layout as _lyra1_checkpoint_layout
from worldarena.models.adapters.lyra2_runner import (
    _checkpoint_tree as _lyra2_checkpoint_tree,
    _trajectory_payload as _lyra2_trajectory_payload,
)
from worldarena.models.adapters.matrix_game import _build_action_payload
from worldarena.models.adapters.worldcam_runner import _existing_base_model_root
from worldarena.models.adapters.worldfm_runner import _build_meta_payload
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


class _FakeVideoApiState:
    def __init__(self) -> None:
        self.submits: list[dict[str, object]] = []
        self.poll_count = 0
        self.task_id = "task-123"
        self.video_bytes = b"fake-api-video"


def _start_fake_video_api_server() -> tuple[HTTPServer, _FakeVideoApiState]:
    state = _FakeVideoApiState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body or "{}")
            state.submits.append({"path": self.path, "body": payload, "headers": dict(self.headers)})
            response = {
                "task_id": state.task_id,
                "status": "submitted",
            }
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == f"/v1/videos/{state.task_id}":
                state.poll_count += 1
                status = "processing" if state.poll_count == 1 else "succeed"
                payload = {
                    "task_id": state.task_id,
                    "status": status,
                    "video_url": f"http://127.0.0.1:{self.server.server_port}/files/{state.task_id}.mp4",
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            if self.path == f"/files/{state.task_id}.mp4":
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(state.video_bytes)))
                self.end_headers()
                self.wfile.write(state.video_bytes)
                return

            self.send_response(404)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


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


def test_matrix_game_2_templerun_payload_omits_mouse_condition() -> None:
    payload = _build_action_payload(
        {
            "mode": "templerun",
            "num_output_frames": 6,
            "action_plan": {
                "default": ["nomove", "jump", "turnleft"],
                "frames_per_action": 4,
            },
        },
        "video_dynamic",
        family="matrix_game_2",
    )
    assert payload["mode"] == "templerun"
    assert payload["num_output_frames"] == 6
    assert payload["total_frames"] == 21
    assert "mouse_condition" not in payload
    assert len(payload["keyboard_condition"]) == 21


def test_infinite_world_action_payload_uses_suite_specific_plan() -> None:
    payload = _build_infinite_world_action_payload(
        {
            "num_chunks": 2,
            "chunk_stride": 20,
            "action_plan": {
                "default": ["forward", "camera_right"],
                "by_suite": {"video_dynamic": ["forward", "forward_right", "camera_left"]},
                "frames_per_action": 8,
            },
        },
        "video_dynamic",
        validation_num_frames=21,
    )
    assert payload["actions"] == ["forward", "forward_right", "camera_left"]
    assert payload["num_chunks"] == 2
    assert payload["required_frames"] == 41
    assert len(payload["frame_actions"]) == 41
    assert payload["frame_actions"][0] == {"move": "go forward", "view": "no-op"}
    assert payload["frame_actions"][8] == {"move": "go forward and go right", "view": "no-op"}
    assert payload["frame_actions"][16] == {"move": "no-op", "view": "turn left"}


def test_matrix_game_1_adapter_uses_dedicated_runner(tmp_path: Path, monkeypatch) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr("worldarena.models.adapters.matrix_game.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    repo_root = tmp_path / "Matrix-Game-1"
    repo_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = tmp_path / "weights"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "matrix_game_1.yaml",
        name="matrix-game-1",
        family="matrix_game_1",
        python_bin=sys.executable,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        entrypoint="inference_bench.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="none",
        reference_mode="first_frame",
        generation={
            "video_length": 65,
            "resolution": [640, 352],
            "bfloat16": True,
            "action_plan": {"default": ["attack"]},
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

    assert len(commands) == 1
    command, cwd, env = commands[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        "worldarena.models.adapters.matrix_game_1_runner",
    ]
    assert command[command.index("--checkpoint_dir") + 1] == str(checkpoint_dir.resolve())
    assert command[command.index("--resolution") + 1 : command.index("--resolution") + 3] == ["640", "352"]
    action_path = Path(command[command.index("--actions_json") + 1])
    payload = json.loads(action_path.read_text(encoding="utf-8"))
    assert payload["variant"] == "matrix_game_1"
    assert len(payload["keyboard_condition"]) == 65
    assert len(payload["mouse_condition"]) == 65
    assert payload["keyboard_condition"][0][5] == 1.0
    assert payload["keyboard_condition"][1][5] == 0.0
    assert payload["keyboard_condition"][8][5] == 1.0
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    assert result["records"][0]["status"] == "generated"


def test_matrix_game_2_adapter_uses_dedicated_runner(tmp_path: Path, monkeypatch) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr("worldarena.models.adapters.matrix_game.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    repo_root = tmp_path / "Matrix-Game-2"
    repo_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = tmp_path / "weights"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "matrix_game_2.yaml",
        name="matrix-game-2",
        family="matrix_game_2",
        python_bin=sys.executable,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        entrypoint="inference.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="none",
        reference_mode="first_frame",
        generation={
            "mode": "universal",
            "num_output_frames": 6,
            "output_fps": 8,
            "action_plan": {
                "default": ["forward", "camera_r"],
                "frames_per_action": 12,
            },
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

    assert len(commands) == 1
    command, cwd, env = commands[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        "worldarena.models.adapters.matrix_game_2_runner",
    ]
    assert command[command.index("--checkpoint_dir") + 1] == str(checkpoint_dir.resolve())
    assert command[command.index("--mode") + 1] == "universal"
    assert command[command.index("--num_output_frames") + 1] == "6"
    assert command[command.index("--output_fps") + 1] == "8"
    action_path = Path(command[command.index("--actions_json") + 1])
    payload = json.loads(action_path.read_text(encoding="utf-8"))
    assert payload["variant"] == "matrix_game_2"
    assert payload["mode"] == "universal"
    assert len(payload["keyboard_condition"]) == 21
    assert len(payload["mouse_condition"]) == 21
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    assert result["records"][0]["status"] == "generated"


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


def test_lyra1_adapter_uses_static_runner(tmp_path: Path, monkeypatch) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr("worldarena.models.adapters.lyra1.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "lyra1.yaml",
        name="lyra-1",
        family="lyra1",
        python_bin=sys.executable,
        repo_root=tmp_path / "Lyra-1",
        checkpoint_dir=tmp_path / "weights",
        entrypoint="sample.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        generation={
            "num_video_frames": 121,
            "fps": 24,
            "multi_trajectory": True,
            "target_index_subsample": 4,
            "output_view_index": 0,
            "static_view_indices_fixed": ["5", "0", "1", "2", "3", "4"],
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

    assert len(commands) == 1
    command, cwd, env = commands[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        "worldarena.models.adapters.lyra1_runner",
    ]
    assert command[command.index("--sample_name") + 1] == "video_static_photorealistic_living_spaces_0001"
    assert command[command.index("--checkpoint_dir") + 1] == str((tmp_path / "weights").resolve())
    assert command[command.index("--static_view_indices_fixed") + 1] == "5,0,1,2,3,4"
    assert "--multi_trajectory" in command
    assert command[command.index("--prompt") + 1] == "A photorealistic indoor living spaces scene."
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
    assert output_path.exists()
    assert result["records"][0]["status"] == "generated"


def test_lyra1_runner_resolves_sibling_cache_layout(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache" / "hfd"
    lyra_dir = cache_root / "Lyra"
    gen3c_dir = cache_root / "nvidia--GEN3C-Cosmos-7B"
    tokenizer_dir = cache_root / "nvidia--Cosmos-Tokenize1-CV8x8x8-720p"
    t5_dir = cache_root / "google-t5--t5-11b"

    lyra_dir.mkdir(parents=True, exist_ok=True)
    gen3c_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    t5_dir.mkdir(parents=True, exist_ok=True)

    (lyra_dir / "lyra_static.pt").write_bytes(b"x")
    (gen3c_dir / "model.pt").write_bytes(b"x")
    for filename in ("encoder.jit", "decoder.jit", "mean_std.pt"):
        (tokenizer_dir / filename).write_bytes(b"x")
    for filename in ("config.json", "spiece.model"):
        (t5_dir / filename).write_bytes(b"x")

    resolved = _lyra1_checkpoint_layout(lyra_dir)

    assert resolved["lyra_dir"] == lyra_dir.resolve()
    assert resolved["gen3c_dir"] == gen3c_dir.resolve()
    assert resolved["tokenizer_dir"] == tokenizer_dir.resolve()
    assert resolved["t5_dir"] == t5_dir.resolve()


def test_lyra2_adapter_uses_runner_and_pose_inputs(tmp_path: Path, monkeypatch) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr("worldarena.models.adapters.lyra2.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        annotation_dir / "poses.npy",
        np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0),
    )
    np.save(
        annotation_dir / "intrinsics.npy",
        np.asarray([[20.0, 24.0, 8.0, 6.0]] * 3, dtype=np.float32),
    )

    sample = BenchmarkSample(
        **{
            **_sample(video_path).to_dict(),
            "annotation_path": str(annotation_dir),
            "has_pose": True,
        }
    )
    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "lyra2.yaml",
        name="lyra-2",
        family="lyra2",
        python_bin=sys.executable,
        repo_root=tmp_path / "Lyra-2",
        checkpoint_dir=tmp_path / "weights",
        entrypoint="lyra_2/_src/inference/lyra2_custom_traj_inference.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        generation={
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "use_moge_scale": False,
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
        "worldarena.models.adapters.lyra2_runner",
    ]
    assert command[command.index("--annotation_path") + 1] == str(annotation_dir.resolve())
    assert command[command.index("--checkpoint_dir") + 1] == str((tmp_path / "weights").resolve())
    assert command[command.index("--sample_name") + 1] == sample.prediction_stem
    assert "--no-use_moge_scale" in command
    assert command[command.index("--prompt") + 1] == "A photorealistic indoor living spaces scene."
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
    assert output_path.exists()
    assert result["records"][0]["status"] == "generated"


def test_lyra2_runner_builds_w2c_trajectory_from_annotations(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 2.0
    np.save(annotation_dir / "poses.npy", poses)
    np.save(
        annotation_dir / "intrinsics.npy",
        np.asarray(
            [
                [20.0, 24.0, 8.0, 6.0],
                [20.0, 24.0, 8.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )

    payload = _lyra2_trajectory_payload(
        annotation_path=str(annotation_dir),
        num_frames=2,
        height=480,
        width=832,
    )

    assert payload["image_height"] == 480
    assert payload["image_width"] == 832
    assert payload["w2c"].shape == (2, 4, 4)
    assert payload["w2c"][1][0][3] == -2.0
    assert payload["intrinsics"].shape == (2, 3, 3)
    assert payload["intrinsics"][0][0][0] == 20.0
    assert payload["intrinsics"][0][1][2] == 6.0


def test_lyra2_runner_accepts_direct_checkpoint_tree(tmp_path: Path) -> None:
    checkpoint_tree = tmp_path / "checkpoints"
    for dirname in ("image_encoder", "lora", "model", "recon", "text_encoder", "vae"):
        (checkpoint_tree / dirname).mkdir(parents=True, exist_ok=True)

    resolved = _lyra2_checkpoint_tree(checkpoint_tree)

    assert resolved == checkpoint_tree.resolve()


def test_worldfm_adapter_uses_runner_and_pose_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    monkeypatch.setattr("worldarena.models.adapters.worldfm.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        annotation_dir / "poses.npy",
        np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0),
    )
    np.save(
        annotation_dir / "intrinsics.npy",
        np.asarray([[20.0, 24.0, 8.0, 6.0]] * 3, dtype=np.float32),
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
        config_path=tmp_path / "worldfm.yaml",
        name="worldfm",
        family="worldfm",
        python_bin=sys.executable,
        repo_root=tmp_path / "worldfm",
        checkpoint_dir=tmp_path / "weights",
        entrypoint="run_pipeline.py",
        output_ext=".mp4",
        supported_suites=["video_static"],
        prompt_mode="none",
        reference_mode="first_frame",
        generation={
            "model_filename": "worldfm_1-step.pth",
            "vae_subdir": "vae",
            "step": 1,
            "image_size": 384,
            "cfg_scale": 3.5,
            "render_size": 256,
            "fps": 12,
            "gpu_index": 1,
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
        "worldarena.models.adapters.worldfm_runner",
    ]
    assert command[command.index("--annotation_path") + 1] == str(annotation_dir.resolve())
    assert command[command.index("--checkpoint_dir") + 1] == str((tmp_path / "weights").resolve())
    assert command[command.index("--model_filename") + 1] == "worldfm_1-step.pth"
    assert command[command.index("--sample_name") + 1] == sample.prediction_stem
    conditioning_image = Path(command[command.index("--conditioning_image") + 1])
    assert conditioning_image.exists()
    assert conditioning_image.suffix == ".png"
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
    assert output_path.exists()
    assert result["records"][0]["status"] == "generated"


def test_worldfm_runner_builds_meta_payload_from_annotations(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 1.5
    np.save(annotation_dir / "poses.npy", poses)
    np.save(
        annotation_dir / "intrinsics.npy",
        np.asarray(
            [
                [20.0, 24.0, 8.0, 6.0],
                [20.0, 24.0, 8.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )

    payload = _build_meta_payload(
        sample_name="sample_001",
        image_name="conditioning.png",
        annotation_path=str(annotation_dir),
    )

    assert payload["name"] == "sample_001"
    assert payload["image"] == "conditioning.png"
    assert payload["K"] == [
        [20.0, 0.0, 8.0],
        [0.0, 24.0, 6.0],
        [0.0, 0.0, 1.0],
    ]
    assert len(payload["c2w"]) == 2
    assert payload["c2w"][1][0][3] == 1.5


def test_hunyuanworld_adapter_uses_runner_and_mesh_archive_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []

    def fake_run_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"PK\x03\x04fake-zip")

    monkeypatch.setattr("worldarena.models.adapters.hunyuanworld.run_command", fake_run_command)

    video_path = tmp_path / "reference.avi"
    _write_video(video_path)
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "caption.json").write_text(
        json.dumps(
            {
                "SceneSummary": "A photorealistic indoor room with a lamp and sofa.",
                "objects": ["lamp", "sofa"],
            }
        ),
        encoding="utf-8",
    )

    sample = _sample(video_path)
    sample = BenchmarkSample(
        **{
            **sample.to_dict(),
            "annotation_path": str(annotation_dir),
            "has_annotation": True,
        }
    )
    model_config = ModelRuntimeConfig(
        config_path=tmp_path / "hunyuanworld.yaml",
        name="hunyuanworld",
        family="hunyuanworld",
        python_bin=sys.executable,
        repo_root=tmp_path / "HunyuanWorld-1.0",
        checkpoint_dir=tmp_path / "cache" / "hfd" / "tencent--HunyuanWorld-1",
        entrypoint="demo_panogen.py",
        output_ext=".zip",
        supported_suites=["video_static"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        generation={
            "seed": 7,
            "gpu_index": 1,
            "classes": "indoor",
            "fp8_attention": True,
            "cache": True,
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
        "worldarena.models.adapters.hunyuanworld_runner",
    ]
    assert command[command.index("--classes") + 1] == "indoor"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--gpu_index") + 1] == "1"
    assert "--fp8_attention" in command
    assert "--cache" in command
    assert command[command.index("--labels_fg1") + 1] == "lamp"
    assert command[command.index("--labels_fg2") + 1] == "sofa"
    conditioning_image = Path(command[command.index("--conditioning_image") + 1])
    assert conditioning_image.exists()
    assert conditioning_image.suffix == ".png"
    assert cwd == Path(__file__).resolve().parents[1]
    assert env is not None
    output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.zip"
    assert output_path.exists()
    assert result["records"][0]["status"] == "generated"
    assert result["records"][0]["scene_class"] == "indoor"
    assert result["records"][0]["label_source"] == "annotation_explicit"


def test_hunyuanworld_adapter_derives_label_groups_from_annotation(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "ann"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "caption.json").write_text(
        json.dumps(
            {
                "SceneSummary": "A bright room with a lamp, sofa, and rug.",
                "objects": ["lamp", "sofa", "rug"],
            }
        ),
        encoding="utf-8",
    )
    sample = BenchmarkSample(
        **{
            **_sample(tmp_path / "reference.avi").to_dict(),
            "annotation_path": str(annotation_dir),
            "has_annotation": True,
        }
    )

    fg1, fg2, label_source = _hunyuanworld_label_groups_for_sample(
        sample,
        generation={"labels_fg1_limit": 1, "labels_fg2_limit": 1},
    )

    assert fg1 == ["lamp"]
    assert fg2 == ["sofa"]
    assert label_source == "annotation_explicit"


def test_video_api_adapter_runs_i2v_polling_pipeline(tmp_path: Path) -> None:
    server, state = _start_fake_video_api_server()
    try:
        video_path = tmp_path / "reference.avi"
        _write_video(video_path)
        model_config = ModelRuntimeConfig(
            config_path=tmp_path / "kling_api.yaml",
            name="kling-api",
            family="kling_api",
            python_bin=sys.executable,
            repo_root=None,
            checkpoint_dir=None,
            entrypoint=None,
            output_ext=".mp4",
            supported_suites=["video_static"],
            prompt_mode="prompt_target",
            reference_mode="first_frame",
            backend_type="http_api",
            task_families=["video_generation"],
            artifact_types=["video"],
            control_signals=["text", "image"],
            roles=["candidate"],
            api={
                "provider": "kling",
                "base_url": f"http://127.0.0.1:{server.server_port}",
                "submit_mode": "http_polling",
                "submit_route": {
                    "t2v": "/v1/videos/text2video",
                    "i2v": "/v1/videos/image2video",
                },
                "status_route": "/v1/videos/{task_id}",
                "task_id_path": "task_id",
                "status_path": "status",
                "artifact_url_path": "video_url",
                "success_statuses": ["succeed"],
                "failure_statuses": ["failed"],
                "headers": {"Content-Type": "application/json"},
                "image_field": "image_url",
                "image_transport": "data_url",
                "poll_interval_seconds": 0.01,
                "timeout_seconds": 5,
            },
            generation={
                "task_type": "auto",
                "model": "kling-v2.6-pro",
                "duration_seconds": 5,
                "aspect_ratio": "16:9",
                "mode": "professional",
            },
        )

        result = generate_predictions(
            model_config=model_config,
            manifest=[_sample(video_path)],
            output_dir=tmp_path / "predictions",
            overwrite=True,
        )

        output_path = tmp_path / "predictions" / "video_static_photorealistic_living_spaces_0001.mp4"
        assert output_path.read_bytes() == state.video_bytes
        assert len(state.submits) == 1
        submit = state.submits[0]
        assert submit["path"] == "/v1/videos/image2video"
        submit_body = submit["body"]
        assert submit_body["prompt"] == "Generate a photorealistic video of a coherent indoor living spaces scene."
        assert submit_body["model"] == "kling-v2.6-pro"
        assert str(submit_body["image_url"]).startswith("data:image/png;base64,")
        record = result["records"][0]
        assert record["status"] == "generated"
        assert record["provider"] == "kling"
        assert record["job_id"] == state.task_id
        assert record["request"]["task_type"] == "i2v"
        assert record["response"]["status"] == "succeed"
    finally:
        server.shutdown()
        server.server_close()


def test_video_api_adapter_can_run_t2v_without_conditioning(tmp_path: Path) -> None:
    server, state = _start_fake_video_api_server()
    try:
        video_path = tmp_path / "reference.avi"
        _write_video(video_path)
        model_config = ModelRuntimeConfig(
            config_path=tmp_path / "text_video_api.yaml",
            name="text-video-api",
            family="text_video_api",
            python_bin=sys.executable,
            repo_root=None,
            checkpoint_dir=None,
            entrypoint=None,
            output_ext=".mp4",
            supported_suites=["video_static"],
            prompt_mode="prompt_target",
            reference_mode="none",
            backend_type="http_api",
            task_families=["video_generation"],
            artifact_types=["video"],
            control_signals=["text"],
            roles=["candidate"],
            api={
                "provider": "generic",
                "base_url": f"http://127.0.0.1:{server.server_port}",
                "submit_mode": "http_polling",
                "submit_route": {
                    "t2v": "/v1/videos/text2video",
                    "i2v": "/v1/videos/image2video",
                },
                "status_route": "/v1/videos/{task_id}",
                "task_id_path": "task_id",
                "status_path": "status",
                "artifact_url_path": "video_url",
                "success_statuses": ["succeed"],
                "failure_statuses": ["failed"],
                "headers": {"Content-Type": "application/json"},
                "poll_interval_seconds": 0.01,
                "timeout_seconds": 5,
            },
            generation={
                "task_type": "auto",
                "model": "generic-t2v",
                "duration_seconds": 8,
            },
        )

        result = generate_predictions(
            model_config=model_config,
            manifest=[_sample(video_path)],
            output_dir=tmp_path / "predictions_t2v",
            overwrite=True,
        )

        submit = state.submits[0]
        assert submit["path"] == "/v1/videos/text2video"
        assert "image_url" not in submit["body"]
        assert result["records"][0]["request"]["task_type"] == "t2v"
    finally:
        server.shutdown()
        server.server_close()
