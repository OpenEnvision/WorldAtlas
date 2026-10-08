from __future__ import annotations

from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.camera_runtime import camera_annotation_for_conditioning_image
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class SanaWMAdapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.sana_wm.sana_wm_batch_runner"
    label = "Sana-WM"

    def batch_checkpoint_load_policy(self) -> str:
        """The batch runner builds SanaWMPipeline once and reuses it for every row."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = self.config.generation
        recipe = generation.get("camera_control_recipe", "legacy")
        if recipe == "official_demo_v1":
            from worldatlas.models.adapters.sana_wm.camera import demo_camera_annotation
            if int(generation.get("fps", 16)) != 16:
                raise ValueError("Sana official demo camera recipe requires 16 fps")
            annotation_path, pose_source, metadata = demo_camera_annotation(
                request.sample, request.conditioning_image.expanduser().resolve(),
                frame_count=int(generation.get("num_frames", 161)),
                focal_scale=float(generation.get("synthetic_focal_scale", .6)),
                repo_root=self.config.repo_root)
            payload["camera_control_recipe"] = metadata
        elif recipe == "legacy":
            annotation_path, pose_source = camera_annotation_for_conditioning_image(
                request.sample,
                request.conditioning_image.expanduser().resolve(),
                frame_count=int(generation.get("num_frames", 161)),
                namespace="sana_wm",
                focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
            )
        else:
            raise ValueError(f"Unknown Sana camera recipe: {recipe}")
        if generation.get("seed_policy") == "per_sample":
            payload["seed"] = int(generation.get("seed", 42))
        payload["record_camera_receipt"] = bool(generation.get("record_camera_receipt", False))
        payload["annotation_path"] = str(annotation_path)
        payload["pose_source"] = pose_source
        return payload
