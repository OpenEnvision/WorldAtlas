"""Core dataclasses shared across benchmark inventory, manifest, and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass()
class DiscoveredAsset:
    """A raw asset discovered during dataset scanning."""
    asset_level: str
    modality: str
    is_formal: bool
    eligible_for_official: bool
    path: str
    relative_path: str
    category_path: str
    source_name: str
    source_type: str
    source_group_id: str
    license_bucket: str
    style: str | None = None
    environment: str | None = None
    scene: str | None = None
    motion_category: str | None = None
    duration_bucket: str | None = None
    fine_class: str | None = None
    annotation_path: str | None = None
    mask_path: str | None = None
    has_annotation: bool = False
    has_instruction: bool = False
    has_pose: bool = False
    has_mask: bool = False
    mask_count: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    probe_error: str | None = None
    usable: bool = True
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """To dict -> dict[str, Any]."""
        return asdict(self)


@dataclass()
class BenchmarkSample:
    """One benchmark sample with metadata and prediction paths."""
    sample_id: str
    suite: str
    split: str
    asset_level: str
    modality: str
    is_formal: bool
    eligible_for_official: bool
    category_path: str
    path: str
    relative_path: str
    reference_path: str
    prediction_stem: str
    conditioning_strategy: str
    prompt_current: str
    prompt_target: str
    style: str | None
    environment: str | None
    scene: str | None
    motion_category: str | None
    source_name: str
    source_type: str
    source_group_id: str
    license_bucket: str
    prompt_sequence: list[str] = field(default_factory=list)
    camera_path: list[str] = field(default_factory=list)
    generation_mode: str | None = None
    benchmark_family: str = "worldarena"
    task_family: str | None = None
    artifact_type: str | None = None
    control_signals: list[str] = field(default_factory=list)
    annotation_path: str | None = None
    mask_path: str | None = None
    has_annotation: bool = False
    has_instruction: bool = False
    has_pose: bool = False
    has_mask: bool = False
    mask_count: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    duration_bucket: str | None = None
    conditioning_path: str | None = None
    conditioning_frame_count: int | None = None
    conditioning_end_ratio: float | None = None
    evaluation_start_ratio: float | None = None
    track: str | None = None
    physics_case: dict[str, Any] | None = None
    physics_spec: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """To dict -> dict[str, Any]."""
        return asdict(self)


@dataclass()
class MetricOutput:
    """The output of a single metric evaluation."""
    raw: Any | None
    normalized: float | None
    backend: str
    details: dict[str, Any] = field(default_factory=dict)
    protocol_name: str | None = None
    eligibility_status: str = "official"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """To dict -> dict[str, Any]."""
        return asdict(self)


@dataclass()
class SampleScore:
    """Per-sample scores with all metric outputs."""
    sample_id: str
    suite: str
    metrics: dict[str, MetricOutput]
    path: str
    prediction_path: str | None
    style: str | None
    environment: str | None
    scene: str | None
    motion_category: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """To dict -> dict[str, Any]."""
        payload = asdict(self)
        payload["metrics"] = {name: output.to_dict() for name, output in self.metrics.items()}
        return payload


__all__ = [
    "BenchmarkSample",
    "DiscoveredAsset",
    "MetricOutput",
    "SampleScore",
]
