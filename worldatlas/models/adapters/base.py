"""Abstract base types for WorldAtlas model adapters.

Each world model implements ``ModelAdapter`` and is invoked by the generation
runner with a ``BenchmarkSample``, conditioning media, and an output path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.models.config import ModelRuntimeConfig


DEFAULT_ROLLOUT_DURATION_SECONDS = 60.0


@dataclass(slots=True)
class PreparedGenerationRequest:
    """One sample queued for generation, with resolved paths and prompt text."""

    sample: BenchmarkSample
    conditioning_image: Path
    output_path: Path
    prompt: str


@dataclass(frozen=True, slots=True)
class RolloutPlan:
    """How many rollout units an adapter must run to cover a target duration.

    Models expose very different rollout geometries — autoregressive iterations,
    latent blocks, per-action clips — so the benchmark fixes wall-clock duration and
    lets each adapter quantize it against its own unit size at its own native frame
    rate. Fixing frames instead would make a 24 fps model cover only half the real
    time span of a 12 fps one.
    """

    target_seconds: float
    native_fps: float
    unit_frames: int
    unit_count: int
    base_frames: int
    truncated: bool

    @property
    def total_frames(self) -> int:
        """Frames the adapter will actually emit under this plan."""
        return self.base_frames + self.unit_frames * self.unit_count

    @property
    def achieved_seconds(self) -> float:
        """Wall-clock duration of the emitted frames at the adapter's native rate."""
        return self.total_frames / max(self.native_fps, 1e-6)

    @property
    def shortfall_seconds(self) -> float:
        """How far the plan falls short of the requested duration, if at all."""
        return max(float(self.target_seconds) - self.achieved_seconds, 0.0)

    def as_details(self) -> dict[str, Any]:
        """Serialize the plan for generation records and downstream auditing."""
        return {
            "rollout_target_seconds": round(float(self.target_seconds), 6),
            "rollout_native_fps": round(float(self.native_fps), 6),
            "rollout_unit_frames": int(self.unit_frames),
            "rollout_unit_count": int(self.unit_count),
            "rollout_base_frames": int(self.base_frames),
            "rollout_total_frames": int(self.total_frames),
            "rollout_achieved_seconds": round(self.achieved_seconds, 6),
            "rollout_truncated": bool(self.truncated),
        }


def plan_rollout(
    *,
    target_seconds: float,
    native_fps: float,
    unit_frames: int,
    base_frames: int = 0,
    min_units: int = 1,
    max_units: int | None = None,
) -> RolloutPlan:
    """Quantize a target duration into whole rollout units, rounding up to cover it.

    Rounding up matters for memory evaluation: a loop trajectory that is cut short
    never reaches its revisit waypoint, which would be scored as forgetting rather
    than as a truncated rollout.
    """
    if unit_frames <= 0:
        raise ValueError(f"unit_frames must be positive, got {unit_frames}")
    if native_fps <= 0.0:
        raise ValueError(f"native_fps must be positive, got {native_fps}")
    if target_seconds <= 0.0:
        raise ValueError(f"target_seconds must be positive, got {target_seconds}")

    needed_frames = target_seconds * native_fps - base_frames
    units = max(int(-(-needed_frames // unit_frames)), int(min_units), 1)
    truncated = False
    if max_units is not None and units > int(max_units):
        units = max(int(max_units), 1)
        truncated = True
    return RolloutPlan(
        target_seconds=float(target_seconds),
        native_fps=float(native_fps),
        unit_frames=int(unit_frames),
        unit_count=int(units),
        base_frames=int(base_frames),
        truncated=truncated,
    )


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

    def rollout_target_seconds(self) -> float | None:
        """Requested rollout duration in seconds, or None to keep the adapter default."""
        value = self.config.generation.get("target_duration_seconds")
        return None if value is None else float(value)

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
