"""Abstract types for remote API video generation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ApiJobHandle:
    provider: str
    job_id: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApiGenerationResult:
    provider: str
    artifact_path: Path
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ApiProvider(ABC):
    """Submit async generation jobs and download finished artifacts."""

    @abstractmethod
    def submit(self, payload: dict[str, Any]) -> ApiJobHandle:
        raise NotImplementedError

    @abstractmethod
    def wait(self, handle: ApiJobHandle) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def download(self, result: dict[str, Any], output_path: Path) -> ApiGenerationResult:
        raise NotImplementedError
