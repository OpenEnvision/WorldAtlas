from __future__ import annotations

from pathlib import Path

from worldarena.models.config import load_model_config


def test_load_model_config_prefers_local_runtime_when_python_bin_is_auto(tmp_path: Path) -> None:
    project_root = tmp_path
    runtime_python = project_root / ".runtime312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    repo_root = project_root / "external" / "lingbot-world"
    repo_root.mkdir(parents=True, exist_ok=True)
    config_path = project_root / "config" / "models" / "lingbot_world.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  name: lingbot-world",
                "  family: lingbot_world",
                "  python_bin: auto",
                f"  repo_root: {repo_root}",
                "  checkpoint_dir: null",
                "  entrypoint: generate.py",
                "  output_ext: .mp4",
                "  supported_suites: [video_static]",
                "  prompt_mode: prompt_target",
                "  reference_mode: first_frame",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_model_config(config_path)
    assert config.python_bin == str(runtime_python.absolute())


def test_load_model_config_keeps_runtime_symlink_path(tmp_path: Path) -> None:
    project_root = tmp_path
    real_python = project_root / "python-real"
    real_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    runtime_python = project_root / ".runtime312" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.symlink_to(real_python)

    repo_root = project_root / "external" / "matrix-game"
    repo_root.mkdir(parents=True, exist_ok=True)
    config_path = project_root / "config" / "models" / "matrix_game.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  name: matrix-game",
                "  family: matrix_game_3",
                "  python_bin: .runtime312/bin/python",
                f"  repo_root: {repo_root}",
                "  checkpoint_dir: null",
                "  entrypoint: generate.py",
                "  output_ext: .mp4",
                "  supported_suites: [video_static]",
                "  prompt_mode: prompt_target",
                "  reference_mode: first_frame",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_model_config(config_path)
    assert config.python_bin == str(runtime_python.absolute())
