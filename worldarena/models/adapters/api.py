from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.providers.base import ApiGenerationResult, ApiProvider


class ApiModelAdapter(ModelAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        if config.backend_type not in {"sdk_api", "http_api"}:
            raise ValueError(
                f"{type(self).__name__} requires an API backend, got {config.backend_type}"
            )

    @abstractmethod
    def build_provider(self) -> ApiProvider:
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
            "request": artifact.request,
            "response": artifact.response,
            "metadata": artifact.metadata,
        }
