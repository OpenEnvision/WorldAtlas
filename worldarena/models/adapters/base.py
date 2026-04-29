from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.config import ModelRuntimeConfig


@dataclass(slots=True)
class PreparedGenerationRequest:
    sample: BenchmarkSample
    conditioning_image: Path
    output_path: Path
    prompt: str


class ModelAdapter(ABC):
    def __init__(self, config: ModelRuntimeConfig) -> None:
        self.config = config

    def supports(self, sample: BenchmarkSample) -> bool:
        return sample.suite in self.config.supported_suites

    def supports_batch_generation(self) -> bool:
        return False

    def prompt_for(self, sample: BenchmarkSample) -> str:
        if self.config.prompt_mode == "none":
            return ""
        if self.config.prompt_mode == "prompt_current":
            prompt = sample.prompt_current
        else:
            prompt = sample.prompt_target
        if not prompt.strip():
            raise ValueError(
                f"{self.config.name} requires non-empty {self.config.prompt_mode} for sample {sample.sample_id}"
            )
        return prompt

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement batch generation")

    @abstractmethod
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError
