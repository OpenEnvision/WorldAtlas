"""HTTP video-generation API adapter (text-to-video and image-to-video).

Maps benchmark samples and generation config onto provider-specific REST payloads
via configurable field names and submit routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.api import ApiModelAdapter
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.common import image_path_to_data_url
from worldarena.models.providers import ApiProvider, HttpPollingProvider


DEFAULT_FIELD_MAP = {
    "prompt": "prompt",
    "negative_prompt": "negative_prompt",
    "model": "model",
    "duration_seconds": "duration",
    "aspect_ratio": "aspect_ratio",
    "mode": "mode",
    "resolution": "resolution",
    "fps": "fps",
    "seed": "seed",
    "callback_url": "callback_url",
    "external_task_id": "external_task_id",
}


class VideoApiAdapter(ApiModelAdapter):
    """Generate videos through a configurable HTTP polling provider."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return super().generate_batch(requests)

    def build_provider(self) -> ApiProvider:
        """Build provider -> ApiProvider."""
        submit_mode = str(self.config.api.get("submit_mode", "http_polling")).strip().lower()
        provider_name = str(self.config.api.get("provider", self.config.name))
        if submit_mode == "http_polling":
            return HttpPollingProvider(provider_name=provider_name, api_config=self.config.api)
        raise ValueError(f"unsupported API submit_mode: {submit_mode}")

    def _task_type(self, sample: BenchmarkSample, conditioning_image: Path) -> str:
        """Infer t2v vs i2v from config, control signals, and conditioning media."""
        explicit = str(
            self.config.generation.get("task_type", self.config.api.get("task_type", "auto"))
        ).strip().lower()
        if explicit in {"t2v", "text2video"}:
            return "t2v"
        if explicit in {"i2v", "image2video"}:
            return "i2v"

        if (
            "image" in self.config.control_signals
            and self.config.reference_mode != "none"
            and conditioning_image.exists()
            and sample.modality == "video"
        ):
            return "i2v"
        return "t2v"

    def _field_map(self) -> dict[str, str]:
        """Field map -> dict[str, str]."""
        field_map = dict(DEFAULT_FIELD_MAP)
        field_map.update(
            {
                str(key): str(value)
                for key, value in dict(self.config.api.get("field_map", {})).items()
            }
        )
        return field_map

    def _image_payload(self, conditioning_image: Path) -> str:
        image_transport = str(self.config.api.get("image_transport", "data_url")).strip().lower()
        if image_transport == "data_url":
            return image_path_to_data_url(conditioning_image)
        if image_transport == "path":
            return str(conditioning_image)
        raise ValueError(f"unsupported API image_transport: {image_transport}")

    def _submit_route(self, task_type: str) -> str:
        submit_route = self.config.api.get("submit_route")
        if isinstance(submit_route, dict):
            if task_type == "i2v":
                route = submit_route.get("i2v") or submit_route.get("image2video")
            else:
                route = submit_route.get("t2v") or submit_route.get("text2video")
            if route:
                return str(route)
        if submit_route:
            return str(submit_route)
        raise ValueError("api.submit_route is required for video API adapters")

    def build_request(
        self,
        *,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        del output_path

        generation = dict(self.config.generation)
        task_type = self._task_type(sample, conditioning_image)
        field_map = self._field_map()
        request_body = {
            str(key): value
            for key, value in dict(self.config.api.get("static_body", {})).items()
        }
        request_body[field_map["prompt"]] = prompt

        for source_key, target_key in field_map.items():
            if source_key == "prompt":
                continue
            value = generation.get(source_key)
            if value is None:
                continue
            request_body[target_key] = value

        request_body.update(
            {
                str(key): value
                for key, value in dict(generation.get("extra_body", {})).items()
            }
        )

        if task_type == "i2v":
            image_field = str(self.config.api.get("image_field", "image_url"))
            request_body[image_field] = self._image_payload(conditioning_image)

        return {
            "submit_route": self._submit_route(task_type),
            "submit_method": str(self.config.api.get("submit_method", "POST")),
            "request_body": request_body,
            "request_timeout_seconds": self.config.api.get("request_timeout_seconds"),
            "task_type": task_type,
        }
