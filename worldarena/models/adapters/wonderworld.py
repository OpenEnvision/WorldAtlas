"""WonderWorld model adapter for WorldAtlas Arena.

WonderWorld builds persistent 3D scenes from a conditioning image and camera
trajectory, with optional pose synthesis for image_static samples.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

from worldarena.benchmark.annotations import derive_object_prompts
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.taxonomy import humanize_label
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)
from worldarena.models.adapters.pose_synthesis import resolve_pose_annotation_path
from worldarena.common.checkpoints import (
    project_root as worldarena_project_root,
    resolve_checkpoint_path,
    resolve_project_path,
)
from worldarena.models.config import ModelRuntimeConfig


def _project_root(config: ModelRuntimeConfig) -> Path:
    del config
    return worldarena_project_root()


def _resolve_extra_path(config: ModelRuntimeConfig, value: str | None) -> Path | None:
    del config
    return resolve_project_path(value)


def _generation_for_suite(generation: dict[str, Any], suite: str) -> dict[str, Any]:
    resolved = dict(generation)
    for key, value in generation.items():
        if not key.endswith("_by_suite") or not isinstance(value, dict):
            continue
        base_key = key[: -len("_by_suite")]
        if suite in value:
            resolved[base_key] = value[suite]
    return resolved


def _synthetic_pose_translation_scale(generation: dict[str, Any]) -> float:
    """Return the synthetic camera translation scale in WonderWorld units."""
    raw_value = generation.get("synthetic_translation_scale", generation.get("camera_speed", 0.001))
    scale = float(raw_value)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "WonderWorld synthetic_translation_scale must be finite and non-negative, "
            f"got {raw_value!r}"
        )
    return scale


def _clean_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _clean_scene_name(value: object | None) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = humanize_label(text)
    return re.sub(r"[.,;:!?]+$", "", text).strip()


def _normalize_entity_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        raw_items = [str(item).strip() for item in value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = _clean_scene_name(item)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _scene_name_for_sample(
    sample: BenchmarkSample,
    *,
    generation: dict[str, Any],
    prompt_text: str,
) -> str:
    explicit = _clean_scene_name(generation.get("scene_name"))
    if explicit:
        return explicit

    scene_text = _clean_scene_name(sample.scene)
    if scene_text:
        return scene_text

    environment_text = _clean_scene_name(sample.environment)
    if environment_text:
        return f"{environment_text} scene"

    prompt_scene = _clean_scene_name(prompt_text)
    if prompt_scene:
        return prompt_scene
    return "scene"


def _entities_for_sample(
    sample: BenchmarkSample,
    *,
    generation: dict[str, Any],
    scene_name: str,
) -> tuple[list[str], str]:
    explicit_entities = _normalize_entity_list(generation.get("entities"))
    if explicit_entities:
        filtered = [
            entity
            for entity in explicit_entities
            if entity.casefold() != scene_name.casefold()
        ]
        return filtered, "config_explicit"

    entity_limit = max(int(generation.get("entity_limit", 3)), 0)
    if entity_limit <= 0:
        return [], "disabled"

    derived_entities, entity_source = derive_object_prompts(
        sample.annotation_path,
        limit=entity_limit,
    )
    filtered: list[str] = []
    seen: set[str] = {scene_name.casefold()}
    for entity in _normalize_entity_list(derived_entities):
        key = entity.casefold()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(entity)
        if len(filtered) >= entity_limit:
            break
    return filtered, entity_source


def _style_prompt_for_sample(
    sample: BenchmarkSample,
    *,
    generation: dict[str, Any],
) -> str:
    explicit = _clean_text(generation.get("style_prompt"))
    if explicit:
        return explicit
    derived = _clean_scene_name(sample.style)
    if derived:
        return derived
    return "photorealistic"


def _structured_prompts_for_sample(
    sample: BenchmarkSample,
    *,
    prompt_text: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    scene_name = _scene_name_for_sample(
        sample,
        generation=generation,
        prompt_text=prompt_text,
    )
    entities, entity_source = _entities_for_sample(
        sample,
        generation=generation,
        scene_name=scene_name,
    )
    content_prompt = ", ".join([scene_name, *entities]) if entities else scene_name
    return {
        "scene_name": scene_name,
        "entities": entities,
        "entity_source": entity_source,
        "style_prompt": _style_prompt_for_sample(sample, generation=generation),
        "background_prompt": _clean_text(generation.get("background_prompt")),
        "negative_prompt": _clean_text(generation.get("negative_prompt")),
        "content_prompt": content_prompt,
    }


def _resolved_repvit_checkpoint(config: ModelRuntimeConfig, generation: dict[str, Any]) -> Path:
    configured_value = generation.get("repvit_checkpoint")
    checkpoint_path = resolve_checkpoint_path(
        configured_value,
        kind="file",
        required=configured_value is not None,
    )
    if checkpoint_path is None:
        raise ValueError("WonderWorld adapter requires generation.repvit_checkpoint")
    return checkpoint_path


def _sequence_from_value(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        value = decoded
    if isinstance(value, list):
        return [" ".join(str(item).split()) for item in value if str(item).strip()]
    return [" ".join(str(value).split())]


def _prompt_sequence_for_sample(
    sample: BenchmarkSample,
    *,
    prompt_text: str,
    structured: dict[str, Any],
    generation: dict[str, Any],
    num_scenes: int,
) -> list[str]:
    configured = _sequence_from_value(generation.get("prompt_sequence"))
    sample_sequence = [" ".join(str(item).split()) for item in sample.prompt_sequence if str(item).strip()]
    scene_prompt = str(structured["content_prompt"] or prompt_text).strip() or "photorealistic scene"

    sequence = configured or sample_sequence
    prompts = [scene_prompt]
    prompts.extend(prompt for prompt in sequence if prompt)
    while len(prompts) < num_scenes + 1:
        prompts.append(prompts[-1])
    return prompts[: num_scenes + 1]


def _resolved_checkpointish_option(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if Path(text).is_absolute() or text.startswith("ckpt/"):
        path = resolve_checkpoint_path(text, kind="any", required=False)
        return str(path) if path is not None else None
    return text


class WonderWorldAdapter(ModelAdapter):
    """Generate persistent 3D world videos via the WonderWorld upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("WonderWorld adapter requires repo_root")

        project_root = _project_root(self.config)
        spec_dir = requests[0].output_path.parent / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"wonderworld_{uuid.uuid4().hex}.jsonl"
        results_path = spec_path.with_suffix(".results.json")
        rows: list[dict[str, Any]] = []
        prepared_results: dict[str, dict[str, Any]] = {}
        metadata_by_sample: dict[str, dict[str, Any]] = {}

        for request in requests:
            sample = request.sample
            output_path = request.output_path
            generation = _generation_for_suite(self.config.generation, sample.suite)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_prefix = str(generation.get("prompt_prefix", ""))
                prompt_suffix = str(generation.get("prompt_suffix", ""))
                prompt_text = f"{prompt_prefix}{request.prompt}{prompt_suffix}".strip()
                if not prompt_text:
                    prompt_text = str(sample.prompt_current or sample.prompt_target or "").strip()

                structured = _structured_prompts_for_sample(
                    sample,
                    prompt_text=prompt_text,
                    generation=generation,
                )
                repvit_checkpoint = _resolved_repvit_checkpoint(self.config, generation)
                num_scenes = max(int(generation.get("num_scenes", 1)), 1)
                trajectory_frames = max(int(generation.get("trajectory_frames", 49)), num_scenes + 1)
                synthetic_translation_scale = _synthetic_pose_translation_scale(generation)
                pose_annotation_path, pose_source = resolve_pose_annotation_path(
                    self.config,
                    sample,
                    request.conditioning_image,
                    frame_count=trajectory_frames,
                    namespace="wonderworld",
                    focal_scale=float(generation.get("synthetic_focal_scale", 1.875)),
                    translation_scale=synthetic_translation_scale,
                )
                prompt_sequence = _prompt_sequence_for_sample(
                    sample,
                    prompt_text=prompt_text,
                    structured=structured,
                    generation=generation,
                    num_scenes=num_scenes,
                )
                archive_path = (
                    output_path.with_name(f"{output_path.stem}.sidecar.zip")
                    if bool(generation.get("archive_sidecars", True))
                    else None
                )

                command = [
                    self.config.python_bin,
                    "-m",
                    "worldarena.models.adapters.wonderworld_batch_runner",
                    "--repo_root",
                    str(repo_root),
                    "--entrypoint",
                    str(self.config.entrypoint or "run.py"),
                    "--conditioning_image",
                    str(request.conditioning_image.expanduser().resolve()),
                    "--annotation_path",
                    str(pose_annotation_path),
                    "--output_path",
                    str(output_path),
                    "--sample_name",
                    output_path.stem,
                    "--prompt_sequence_json",
                    json.dumps(prompt_sequence, ensure_ascii=False),
                    "--style_prompt",
                    structured["style_prompt"],
                    "--repvit_checkpoint",
                    str(repvit_checkpoint),
                    "--num_scenes",
                    str(num_scenes),
                    "--trajectory_frames",
                    str(trajectory_frames),
                ]
                if archive_path is not None:
                    command.extend(["--archive_path", str(archive_path)])
                stable_diffusion_checkpoint = _resolved_checkpointish_option(
                    generation.get("stable_diffusion_checkpoint", "sd2-community/stable-diffusion-2-inpainting")
                )
                if stable_diffusion_checkpoint is not None:
                    command.extend(["--stable_diffusion_checkpoint", stable_diffusion_checkpoint])
                for option in ("depth_model_repo", "normal_model_repo", "oneformer_model_repo"):
                    value = _resolved_checkpointish_option(generation.get(option))
                    if value is not None:
                        command.extend([f"--{option}", value])

                extend_command_with_options(
                    command,
                    payload=generation,
                    value_options={
                        "seed": int,
                        "fps": float,
                        "depth_model": str,
                        "camera_speed": float,
                        "fg_depth_range": float,
                        "depth_shift": float,
                        "sky_hard_depth": float,
                        "init_focal_length": int,
                        "inpainting_resolution_gen": int,
                        "inpainting_resolution_interp": int,
                        "rotation_path": str,
                        "min_render_content_fraction": float,
                        "max_render_blank_frame_ratio": float,
                    },
                    bool_value_options=(
                        "use_gpt",
                        "debug",
                        "gen_layer",
                        "use_compile",
                        "keep_work_dir",
                    ),
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "args": command[3:],
                        "prediction_path": str(output_path),
                        "prompt": prompt_text,
                    }
                )
                metadata_by_sample[sample.sample_id] = {
                    "command": command,
                    "prediction_path": str(output_path),
                    "prompt": prompt_text,
                    "scene_name": structured["scene_name"],
                    "entities": structured["entities"],
                    "entity_source": structured["entity_source"],
                    "style_prompt": structured["style_prompt"],
                    "background_prompt": structured["background_prompt"],
                    "negative_prompt": structured["negative_prompt"],
                    "content_prompt": structured["content_prompt"],
                    "prompt_sequence": prompt_sequence,
                    "pose_annotation_path": str(pose_annotation_path),
                    "pose_source": pose_source,
                    "synthetic_pose_translation_scale": synthetic_translation_scale,
                }
                if archive_path is not None:
                    metadata_by_sample[sample.sample_id]["archive_path"] = str(archive_path)
            except Exception as exc:
                prepared_results[sample.sample_id] = {
                    "status": "failed",
                    "error": str(exc),
                    "prediction_path": str(output_path),
                    "prompt": request.prompt,
                }

        loaded_results: dict[str, dict[str, Any]] = {}
        if rows:
            with spec_path.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False))
                    file.write(chr(10))
            command = [
                self.config.python_bin,
                "-m",
                "worldarena.models.adapters.wonderworld_batch_runner",
                "--batch_spec",
                str(spec_path),
                "--batch_results",
                str(results_path),
            ]
            env = build_env(extra_pythonpaths=[project_root], overrides=self.config.env)
            run_command(command, cwd=project_root, env=env)
            if results_path.exists():
                loaded_results = json.loads(results_path.read_text(encoding="utf-8"))

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            sample_id = request.sample.sample_id
            payload = dict(prepared_results.get(sample_id) or loaded_results.get(sample_id) or {})
            if not payload:
                payload = {
                    "status": "failed",
                    "error": "batch generation returned no result for this sample",
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                }
            if payload.get("status") == "generated":
                payload.update(metadata_by_sample.get(sample_id, {}))
                frames_stem = request.output_path.with_suffix("")
                frames_dir = frames_stem.with_name(f"{frames_stem.name}.frames")
                if frames_dir.exists():
                    payload["frames_dir"] = str(frames_dir)
            results[sample_id] = payload
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("WonderWorld adapter requires repo_root")

        generation = _generation_for_suite(self.config.generation, sample.suite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = _project_root(self.config)

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()
        if not prompt_text:
            prompt_text = str(sample.prompt_current or sample.prompt_target or "").strip()

        structured = _structured_prompts_for_sample(
            sample,
            prompt_text=prompt_text,
            generation=generation,
        )
        repvit_checkpoint = _resolved_repvit_checkpoint(self.config, generation)
        num_scenes = max(int(generation.get("num_scenes", 1)), 1)
        trajectory_frames = max(int(generation.get("trajectory_frames", 49)), num_scenes + 1)
        synthetic_translation_scale = _synthetic_pose_translation_scale(generation)
        pose_annotation_path, pose_source = resolve_pose_annotation_path(
            self.config,
            sample,
            conditioning_image,
            frame_count=trajectory_frames,
            namespace="wonderworld",
            focal_scale=float(generation.get("synthetic_focal_scale", 1.875)),
            translation_scale=synthetic_translation_scale,
        )
        prompt_sequence = _prompt_sequence_for_sample(
            sample,
            prompt_text=prompt_text,
            structured=structured,
            generation=generation,
            num_scenes=num_scenes,
        )
        archive_path = (
            output_path.with_name(f"{output_path.stem}.sidecar.zip")
            if bool(generation.get("archive_sidecars", True))
            else None
        )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.wonderworld_batch_runner",
            "--repo_root",
            str(repo_root),
            "--entrypoint",
            str(self.config.entrypoint or "run.py"),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--annotation_path",
            str(pose_annotation_path),
            "--output_path",
            str(output_path),
            "--sample_name",
            output_path.stem,
            "--prompt_sequence_json",
            json.dumps(prompt_sequence, ensure_ascii=False),
            "--repvit_checkpoint",
            str(repvit_checkpoint),
            "--num_scenes",
            str(num_scenes),
            "--trajectory_frames",
            str(trajectory_frames),
        ]
        if archive_path is not None:
            command.extend(["--archive_path", str(archive_path)])
        stable_diffusion_checkpoint = _resolved_checkpointish_option(
            generation.get("stable_diffusion_checkpoint", "sd2-community/stable-diffusion-2-inpainting")
        )
        if stable_diffusion_checkpoint is not None:
            command.extend(["--stable_diffusion_checkpoint", stable_diffusion_checkpoint])
        for option in ("depth_model_repo", "normal_model_repo"):
            value = _resolved_checkpointish_option(generation.get(option))
            if value is not None:
                command.extend([f"--{option}", value])

        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "seed": int,
                "fps": float,
                "depth_model": str,
                "camera_speed": float,
                "fg_depth_range": float,
                "depth_shift": float,
                "sky_hard_depth": float,
                "init_focal_length": float,
                "inpainting_resolution_gen": int,
                "inpainting_resolution_interp": int,
                "rotation_path": str,
                "min_render_content_fraction": float,
                "max_render_blank_frame_ratio": float,
            },
            bool_value_options=(
                "use_gpt",
                "debug",
                "gen_layer",
                "use_compile",
                "keep_work_dir",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"WonderWorld output was not written: {output_path}")
        payload = {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "scene_name": structured["scene_name"],
            "entities": structured["entities"],
            "entity_source": structured["entity_source"],
            "style_prompt": structured["style_prompt"],
            "background_prompt": structured["background_prompt"],
            "negative_prompt": structured["negative_prompt"],
            "content_prompt": structured["content_prompt"],
            "prompt_sequence": prompt_sequence,
            "pose_annotation_path": str(pose_annotation_path),
            "pose_source": pose_source,
            "synthetic_pose_translation_scale": synthetic_translation_scale,
        }
        if archive_path is not None and archive_path.exists():
            payload["archive_path"] = str(archive_path)
        frames_stem = output_path.with_suffix("")
        frames_dir = frames_stem.with_name(f"{frames_stem.name}.frames")
        if frames_dir.exists():
            payload["frames_dir"] = str(frames_dir)
        return payload


__all__ = [
    "WonderWorldAdapter",
    "_entities_for_sample",
    "_generation_for_suite",
    "_scene_name_for_sample",
    "_structured_prompts_for_sample",
]
