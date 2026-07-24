"""Abstract base types for WorldAtlas Arena model adapters.

Each world model implements ``ModelAdapter`` and is invoked by the generation
runner with a ``BenchmarkSample``, conditioning media, and an output path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.config import ModelRuntimeConfig


@dataclass(slots=True)
class PreparedGenerationRequest:
    """One sample queued for generation, with resolved paths and prompt text."""

    sample: BenchmarkSample
    conditioning_image: Path
    output_path: Path
    prompt: str


class ModelAdapter(ABC):
    """Interface that every benchmarked world model must implement."""

    def __init__(self, config: ModelRuntimeConfig) -> None:
        self.config = config

    def supports(self, sample: BenchmarkSample) -> bool:
        """Return True when this adapter is configured for the sample's suite."""
        return sample.suite in self.config.supported_suites

    def supports_batch_generation(self) -> bool:
        """Override to True when the adapter can amortize model load across samples."""
        return False

    def batch_checkpoint_load_policy(self) -> str:
        """Report how checkpoints are loaded during ``generate_batch``.

        ``load_once`` is reserved for a verified batch worker that initializes
        model weights once and then processes all requests with that resident
        runtime.  A batch-shaped API alone is not enough to claim this: some
        legacy adapters still launch one single-sample process per request.
        """
        if not self.supports_batch_generation():
            return "unsupported"
        return "unverified"

    def prompt_for(self, sample: BenchmarkSample) -> str:
        """Resolve the text prompt according to ``config.prompt_mode``."""
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
        """Generate predictions for multiple samples; default raises NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement batch generation")

    @abstractmethod
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        """Run inference for a single sample and return metadata about the run."""
        raise NotImplementedError
