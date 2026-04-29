from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

import yaml

from worldarena.common.checkpoints import apply_checkpoint_env


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena WonderWorld subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--entrypoint", default="run.py", type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--scene_name", required=True, type=str)
    parser.add_argument("--entities", nargs="*", default=None)
    parser.add_argument("--style_prompt", default="photorealistic", type=str)
    parser.add_argument("--background_prompt", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--repvit_checkpoint", required=True, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--port", default=0, type=int)
    parser.add_argument("--startup_timeout", default=120.0, type=float)
    parser.add_argument("--save_timeout", default=10800.0, type=float)
    parser.add_argument("--save_retry_interval", default=5.0, type=float)
    parser.add_argument("--depth_model", default="marigold", type=str)
    parser.add_argument("--camera_speed", default=0.001, type=float)
    parser.add_argument("--fg_depth_range", default=0.015, type=float)
    parser.add_argument("--depth_shift", default=0.001, type=float)
    parser.add_argument("--sky_hard_depth", default=0.02, type=float)
    parser.add_argument("--init_focal_length", default=960.0, type=float)
    parser.add_argument("--use_gpt", default=False, type=_parse_bool)
    parser.add_argument("--debug", default=False, type=_parse_bool)
    parser.add_argument("--gen_sky_image", default=False, type=_parse_bool)
    parser.add_argument("--gen_sky", default=False, type=_parse_bool)
    parser.add_argument("--gen_layer", default=False, type=_parse_bool)
    parser.add_argument("--load_gen", default=False, type=_parse_bool)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    return parser.parse_args()


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_wonderworld_"))
        try:
            yield path
        finally:
            pass
        return

    with tempfile.TemporaryDirectory(prefix="worldarena_wonderworld_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def _normalized_env(repo_root: Path) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _require_repo_layout(repo_root: Path, entrypoint: str) -> tuple[Path, Path]:
    entrypoint_path = (repo_root / entrypoint).resolve()
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"WonderWorld entrypoint not found: {entrypoint_path}")
    base_config_path = (repo_root / "config" / "base-config.yaml").resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"WonderWorld base config not found: {base_config_path}")
    return entrypoint_path, base_config_path


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _pick_port(port: int) -> int:
    if port > 0:
        return int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _wait_for_server(*, port: int, timeout: float, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + max(float(timeout), 1.0)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "WonderWorld exited before the SocketIO server became ready "
                f"(exit_code={process.returncode})"
            )
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for WonderWorld server on port {port}")


def _connect_client(port: int, *, timeout: float):
    try:
        import socketio as socketio_client
    except ImportError as exc:  # pragma: no cover - depends on external env
        raise ImportError(
            "WonderWorld runner requires python-socketio in the selected Python environment."
        ) from exc

    deadline = time.monotonic() + max(float(timeout), 1.0)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = socketio_client.Client(
            reconnection=False,
            logger=False,
            engineio_logger=False,
        )
        try:
            client.connect(
                f"http://127.0.0.1:{int(port)}",
                wait=True,
                wait_timeout=5.0,
                transports=["polling"],
            )
            return client
        except Exception as exc:  # pragma: no cover - external server timing
            last_error = exc
            with contextlib.suppress(Exception):
                client.disconnect()
            time.sleep(0.5)
    raise RuntimeError(f"Timed out connecting to WonderWorld SocketIO server: {last_error}")


def _write_example_yaml(
    path: Path,
    *,
    example_name: str,
    image_filename: str,
    style_prompt: str,
    content_prompt: str,
    negative_prompt: str,
    background_prompt: str,
) -> Path:
    payload = [
        {
            "name": example_name,
            "image_filepath": f"examples/images/{image_filename}",
            "style_prompt": style_prompt,
            "content_prompt": content_prompt,
            "negative_prompt": negative_prompt,
            "background": background_prompt,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _write_example_config(path: Path, *, args: argparse.Namespace, example_name: str) -> Path:
    payload = {
        "runs_dir": f"output/{example_name}",
        "example_name": example_name,
        "seed": int(args.seed),
        "use_gpt": bool(args.use_gpt),
        "debug": bool(args.debug),
        "depth_model": str(args.depth_model),
        "camera_speed": float(args.camera_speed),
        "fg_depth_range": float(args.fg_depth_range),
        "depth_shift": float(args.depth_shift),
        "sky_hard_depth": float(args.sky_hard_depth),
        "init_focal_length": float(args.init_focal_length),
        "gen_sky_image": bool(args.gen_sky_image),
        "gen_sky": bool(args.gen_sky),
        "gen_layer": bool(args.gen_layer),
        "load_gen": bool(args.load_gen),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _content_prompt(scene_name: str, entities: list[str]) -> str:
    values = [scene_name, *entities]
    return ", ".join(value for value in values if value)


def _expected_artifact_paths(example_dir: Path, example_name: str) -> dict[str, Path]:
    return {
        "point_cloud": example_dir / "finished_3dgs.ply",
        "splat": example_dir / f"{example_name}_finished_3dgs.splat",
        "visibility_filter": example_dir / "visibility_filter_all.pth",
        "sky_filter": example_dir / "is_sky_filter.pth",
        "delete_mask": example_dir / "delete_mask_all.pth",
        "sky_ply": example_dir / "finished_3dgs_sky_tanh.ply",
        "sky_0": example_dir / "sky_0.png",
        "sky_1": example_dir / "sky_1.png",
        "sky_2": example_dir / "sky_2.png",
    }


def _latest_run_dir(runs_root: Path) -> Path | None:
    candidates = [path for path in runs_root.glob("Gen-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _archive_prediction(
    *,
    output_path: Path,
    work_dir: Path,
    example_name: str,
    conditioning_image: Path,
    example_yaml_path: Path,
    example_config_path: Path,
    artifact_paths: dict[str, Path],
    prompt: str,
    scene_name: str,
    entities: list[str],
    style_prompt: str,
    background_prompt: str,
    negative_prompt: str,
    port: int,
    keep_work_dir: bool,
) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    runs_root = work_dir / "output" / example_name
    latest_run_dir = _latest_run_dir(runs_root)
    archive_members: list[str] = []

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def _write_file(source: Path, member: str) -> None:
            if not source.exists() or not source.is_file():
                return
            archive.write(source, arcname=member)
            archive_members.append(member)

        _write_file(conditioning_image, "inputs/conditioning_image" + conditioning_image.suffix)
        _write_file(example_yaml_path, "inputs/examples.yaml")
        _write_file(example_config_path, "config/worldarena_example.yaml")

        for key, path in artifact_paths.items():
            if path.exists():
                _write_file(path, f"artifacts/{path.name}")

        sky_2_path = work_dir / "sky_img" / example_name / "sky_2.png"
        if sky_2_path.exists():
            _write_file(sky_2_path, "artifacts/sky_2.png")

        if latest_run_dir is not None:
            for filename in ("rendered_img.png", "config.yaml", "log.txt"):
                _write_file(latest_run_dir / filename, f"run/{filename}")

        manifest = {
            "prompt": prompt,
            "scene_name": scene_name,
            "entities": entities,
            "content_prompt": _content_prompt(scene_name, entities),
            "style_prompt": style_prompt,
            "background_prompt": background_prompt,
            "negative_prompt": negative_prompt,
            "example_name": example_name,
            "conditioning_image": str(conditioning_image),
            "port": int(port),
            "keep_work_dir": bool(keep_work_dir),
            "work_dir": str(work_dir) if keep_work_dir else None,
            "artifacts": {
                key: path.name
                for key, path in artifact_paths.items()
                if path.exists()
            },
            "run_dir": str(latest_run_dir) if latest_run_dir is not None else None,
        }
        archive.writestr(
            "worldarena_wonderworld_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        archive_members.append("worldarena_wonderworld_manifest.json")
    return archive_members


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.wait(timeout=30)


def _wait_for_artifacts(
    *,
    client,
    process: subprocess.Popen[bytes],
    artifact_paths: dict[str, Path],
    timeout: float,
    retry_interval: float,
) -> None:
    required_paths = [
        artifact_paths["point_cloud"],
        artifact_paths["splat"],
    ]
    deadline = time.monotonic() + max(float(timeout), 1.0)
    last_emit = 0.0
    while time.monotonic() < deadline:
        if all(path.exists() for path in required_paths):
            return
        if process.poll() is not None:
            missing = [str(path) for path in required_paths if not path.exists()]
            raise RuntimeError(
                "WonderWorld exited before writing required artifacts: "
                + ", ".join(missing)
                + f" (exit_code={process.returncode})"
            )
        now = time.monotonic()
        if now - last_emit >= max(float(retry_interval), 1.0):
            try:
                client.emit("save")
            except Exception:  # pragma: no cover - external server timing
                pass
            last_emit = now
        time.sleep(0.5)
    missing = [str(path) for path in required_paths if not path.exists()]
    raise TimeoutError(
        "Timed out waiting for WonderWorld artifacts: " + ", ".join(missing)
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    repvit_checkpoint = Path(args.repvit_checkpoint).expanduser().resolve()

    if not conditioning_image.exists():
        raise FileNotFoundError(f"WonderWorld conditioning image not found: {conditioning_image}")
    if not repvit_checkpoint.exists():
        raise FileNotFoundError(f"WonderWorld RepViT checkpoint not found: {repvit_checkpoint}")

    entrypoint_path, base_config_path = _require_repo_layout(repo_root, args.entrypoint)
    example_name = _slugify(args.sample_name)
    entities = [str(value) for value in (args.entities or []) if str(value).strip()]
    content_prompt = _content_prompt(args.scene_name, entities)

    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        examples_root = work_dir / "examples"
        image_suffix = conditioning_image.suffix or ".png"
        local_conditioning = examples_root / "images" / f"{example_name}{image_suffix}"
        local_conditioning.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conditioning_image, local_conditioning)
        (examples_root / "sky_images" / example_name).mkdir(parents=True, exist_ok=True)
        (work_dir / "output").mkdir(parents=True, exist_ok=True)
        _link_or_copy(repvit_checkpoint, work_dir / "repvit_sam.pt")

        example_yaml_path = _write_example_yaml(
            examples_root / "examples.yaml",
            example_name=example_name,
            image_filename=local_conditioning.name,
            style_prompt=str(args.style_prompt),
            content_prompt=content_prompt,
            negative_prompt=str(args.negative_prompt),
            background_prompt=str(args.background_prompt),
        )
        example_config_path = _write_example_config(
            work_dir / "config" / "worldarena_example.yaml",
            args=args,
            example_name=example_name,
        )

        port = _pick_port(int(args.port))
        command = [
            sys.executable,
            str(entrypoint_path),
            "--base-config",
            str(base_config_path),
            "--example_config",
            str(example_config_path),
            "--port",
            str(port),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(work_dir),
            env=_normalized_env(repo_root),
            start_new_session=True,
        )

        client = None
        try:
            _wait_for_server(
                port=port,
                timeout=float(args.startup_timeout),
                process=process,
            )
            client = _connect_client(port, timeout=float(args.startup_timeout))
            artifact_paths = _expected_artifact_paths(
                work_dir / "examples" / "sky_images" / example_name,
                example_name,
            )
            _wait_for_artifacts(
                client=client,
                process=process,
                artifact_paths=artifact_paths,
                timeout=float(args.save_timeout),
                retry_interval=float(args.save_retry_interval),
            )
            archive_members = _archive_prediction(
                output_path=output_path,
                work_dir=work_dir,
                example_name=example_name,
                conditioning_image=local_conditioning,
                example_yaml_path=example_yaml_path,
                example_config_path=example_config_path,
                artifact_paths=artifact_paths,
                prompt=str(args.prompt),
                scene_name=str(args.scene_name),
                entities=entities,
                style_prompt=str(args.style_prompt),
                background_prompt=str(args.background_prompt),
                negative_prompt=str(args.negative_prompt),
                port=port,
                keep_work_dir=bool(args.keep_work_dir),
            )
            if not archive_members:
                raise FileNotFoundError(f"WonderWorld archive is empty: {output_path}")
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    client.disconnect()
            _terminate_process(process)


if __name__ == "__main__":
    main()
