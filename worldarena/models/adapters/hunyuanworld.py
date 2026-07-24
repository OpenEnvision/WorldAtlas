"""HunyuanWorld model adapter for WorldAtlas Arena.

Builds scene/object label groups from sample annotations and launches
``hunyuanworld_runner`` for panoramic world generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.annotations import derive_object_prompts
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, run_command, run_sequential_generate_batch


def _normalize_label_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        raw_items = [str(item).strip() for item in value if str(item).strip()]

    normalized: list[str] = []
    for item in raw_items:
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _scene_class_for_sample(
    sample: BenchmarkSample,
    *,
    override: str | None = None,
) -> str:
    if override:
        return str(override).strip()
    environment = (sample.environment or "").strip().lower()
    if environment == "indoor":
        return "indoor"
    return "outdoor"


def _label_groups_for_sample(
    sample: BenchmarkSample,
    *,
    generation: dict[str, Any],
) -> tuple[list[str], list[str], str]:
    explicit_fg1 = _normalize_label_list(generation.get("labels_fg1"))
    explicit_fg2 = _normalize_label_list(generation.get("labels_fg2"))
    if explicit_fg1 or explicit_fg2:
        return explicit_fg1, explicit_fg2, "config_explicit"

    fg1_limit = max(int(generation.get("labels_fg1_limit", 1)), 0)
    fg2_limit = max(int(generation.get("labels_fg2_limit", 1)), 0)
    derived_limit = max(
        int(generation.get("derived_label_limit", fg1_limit + fg2_limit)),
        fg1_limit + fg2_limit,
    )
    if derived_limit <= 0:
        return [], [], "disabled"

    derived_labels, label_source = derive_object_prompts(
        sample.annotation_path,
        limit=derived_limit,
    )
    fg1_labels = derived_labels[:fg1_limit]
    fg2_labels = derived_labels[fg1_limit : fg1_limit + fg2_limit]
    return fg1_labels, fg2_labels, label_source


class HunyuanWorldAdapter(ModelAdapter):
    """Generate panoramic world videos via the HunyuanWorld upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "reload_per_sample"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return run_sequential_generate_batch(self, requests)

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("HunyuanWorld adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("HunyuanWorld adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()
        input_mode = "image_to_world" if conditioning_image.exists() else "text_to_world"
        if input_mode == "text_to_world" and not prompt_text:
            raise ValueError("HunyuanWorld requires either a conditioning image or a non-empty prompt")

        scene_class = _scene_class_for_sample(
            sample,
            override=generation.get("classes"),
        )
        labels_fg1, labels_fg2, label_source = _label_groups_for_sample(
            sample,
            generation=generation,
        )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.hunyuanworld_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt_text,
            "--negative_prompt",
            str(generation.get("negative_prompt", "")),
            "--classes",
            scene_class,
            "--seed",
            str(int(generation.get("seed", 42))),
            "--gpu_index",
            str(int(generation.get("gpu_index", 0))),
        ]
        if conditioning_image.exists():
            command.extend(["--conditioning_image", str(conditioning_image.expanduser().resolve())])
        if sample.annotation_path:
            command.extend(
                [
                    "--annotation_path",
                    str(Path(sample.annotation_path).expanduser().resolve()),
                ]
            )
        if labels_fg1:
            command.extend(["--labels_fg1", *labels_fg1])
        if labels_fg2:
            command.extend(["--labels_fg2", *labels_fg2])
        for flag_name in ("fp8_attention", "fp8_gemm", "cache", "export_drc"):
            if generation.get(flag_name, False):
                command.append(f"--{flag_name}")

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"HunyuanWorld output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "scene_class": scene_class,
            "labels_fg1": labels_fg1,
            "labels_fg2": labels_fg2,
            "label_source": label_source,
            "input_mode": input_mode,
        }


__all__ = [
    "HunyuanWorldAdapter",
    "_label_groups_for_sample",
    "_scene_class_for_sample",
]
