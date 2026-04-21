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


def test_load_model_config_parses_optional_env_overrides(tmp_path: Path) -> None:
    project_root = tmp_path
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
                "  python_bin: python",
                f"  repo_root: {repo_root}",
                "  checkpoint_dir: null",
                "  entrypoint: generate.py",
                "  output_ext: .mp4",
                "  supported_suites: [video_static]",
                "  prompt_mode: prompt_target",
                "  reference_mode: first_frame",
                "  env:",
                "    CUDA_VISIBLE_DEVICES: '2'",
                "    HF_HOME: /tmp/hf-cache",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_model_config(config_path)
    assert config.env == {
        "CUDA_VISIBLE_DEVICES": "2",
        "HF_HOME": "/tmp/hf-cache",
    }


def test_repository_model_configs_use_project_relative_layout() -> None:
    project_root = Path(__file__).resolve().parents[1]

    lingbot = load_model_config(project_root / "config" / "models" / "lingbot_world.yaml")
    matrix_game = load_model_config(project_root / "config" / "models" / "matrix_game.yaml")

    assert lingbot.repo_root == (project_root / "third_party" / "lingbot-world").resolve()
    assert lingbot.checkpoint_dir == (
        project_root / "cache" / "hfd" / "robbyant--lingbot-world-base-cam"
    ).resolve()
    assert matrix_game.repo_root == (project_root / "third_party" / "Matrix-Game-3.0").resolve()
    assert matrix_game.checkpoint_dir == (project_root / "cache" / "hfd" / "Matrix-Game-3.0").resolve()
