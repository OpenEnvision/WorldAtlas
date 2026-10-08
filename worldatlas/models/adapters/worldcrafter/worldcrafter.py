"""WorldCrafter image and long-sequence adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.camera_runtime import camera_annotation_for_conditioning_image
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter
from worldatlas.models.adapters.worldcrafter.geometry import output_frame_count, rollout_geometry


class WorldCrafterAdapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.worldcrafter.worldcrafter_batch_runner"
    label = "WorldCrafter"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        command = super()._build_command(spec_path, generation_path)
        binary = Path(command[0])
        if binary.parent.name == "bin":
            # Enter a symlinked venv through its real directory so Python imports
            # large Torch packages from the local filesystem on cluster nodes.
            real_env = binary.parent.parent.resolve()
            real_binary = real_env / "bin" / binary.name
            if real_binary.is_file():
                command[0] = str(real_binary)
        return command

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        chunks, frames, _ = rollout_geometry(self.config.generation)
        output_frames = output_frame_count(
            str(self.config.generation.get("model_type", "fast")), frames,
        )
        recipe = self.config.generation.get("camera_control_recipe", "legacy")
        if recipe == "official_demo_v1":
            from worldatlas.models.adapters.worldcrafter.camera import demo_camera_annotation
            if int(self.config.generation.get("fps", 16)) != 16:
                raise ValueError("WorldCrafter official demo camera recipe requires 16 fps")
            annotation, pose_source, camera_recipe = demo_camera_annotation(
                request.sample, frame_count=frames, video_frame_count=output_frames)
            payload["camera_control_recipe"] = camera_recipe
        elif recipe == "legacy":
            annotation, pose_source = camera_annotation_for_conditioning_image(
                request.sample,
                request.conditioning_image.expanduser().resolve(),
                frame_count=frames,
                namespace="worldcrafter",
            )
        else:
            raise ValueError(f"Unknown WorldCrafter camera recipe: {recipe}")
        payload.update(
            annotation_path=str(annotation),
            pose_source=pose_source,
            num_chunks=chunks,
            num_frames=frames,
            output_frame_count=output_frames,
            achieved_duration_seconds=output_frames / int(self.config.generation.get("fps", 16)),
        )
        return payload
