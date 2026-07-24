"""Abstract base for remote API-backed world model adapters.

API adapters build a provider-specific request, submit it asynchronously,
poll until completion, and download the generated artifact into the predictions tree.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import run_sequential_generate_batch
from worldarena.models.providers.base import ApiGenerationResult, ApiProvider


class ApiModelAdapter(ModelAdapter):
    """Submit generation jobs to an external HTTP or SDK API instead of a local repo."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if config.backend_type not in {"sdk_api", "http_api"}:
            raise ValueError(
                f"{type(self).__name__} requires an API backend, got {config.backend_type}"
            )

    @abstractmethod
    def build_provider(self) -> ApiProvider:
        """Build provider -> ApiProvider."""
        raise NotImplementedError

    @abstractmethod
    def build_request(
        self,
        *,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

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
        provider = self.build_provider()
        request = self.build_request(
            sample=sample,
            conditioning_image=conditioning_image,
            output_path=output_path,
            prompt=prompt,
        )
        handle = provider.submit(request)
        result = provider.wait(handle)
        artifact = provider.download(result, output_path)
        if not artifact.request:
            artifact.request = dict(request)
        return self._to_record(handle=handle, artifact=artifact)

    def _to_record(
        self,
        *,
        handle,
        artifact: ApiGenerationResult,
    ) -> dict[str, Any]:
        return {
            "prediction_path": str(artifact.artifact_path),
            "provider": artifact.provider,
            "job_id": handle.job_id,
            "submit_response": handle.raw,
            "request": artifact.request,
            "response": artifact.response,
            "metadata": artifact.metadata,
        }
