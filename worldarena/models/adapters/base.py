from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from worldarena.models.config import ModelRuntimeConfig
from worldarena.benchmark.schemas import BenchmarkSample


class ModelAdapter(ABC):
    def __init__(self, config: ModelRuntimeConfig) -> None:
        self.config = config

    def supports(self, sample: BenchmarkSample) -> bool:
        return sample.suite in self.config.supported_suites

    def prompt_for(self, sample: BenchmarkSample) -> str:
        if self.config.prompt_mode == "none":
            return ""
        if self.config.prompt_mode == "prompt_current":
            return sample.prompt_current
        return sample.prompt_target

    @abstractmethod
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError
