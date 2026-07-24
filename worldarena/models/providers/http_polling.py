"""HTTP polling provider for remote video generation APIs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from worldarena.models.providers.base import ApiGenerationResult, ApiJobHandle, ApiProvider


def _resolve_url(*, base_url: str | None, route_or_url: str) -> str:
    route = str(route_or_url).strip()
    if route.startswith(("http://", "https://")):
        return route
    if not base_url:
        raise ValueError(f"base_url is required for relative route {route!r}")
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


def _should_bypass_proxies(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _urlopen(request: Request, *, timeout_seconds: float):
    if _should_bypass_proxies(request.full_url):
        opener = build_opener(ProxyHandler({}))
        return opener.open(request, timeout=timeout_seconds)
    return urlopen(request, timeout=timeout_seconds)


def _extract_path(payload: Any, path: str | None) -> Any:
    if not path:
        return None
    current = payload
    for raw_token in str(path).split("."):
        token = raw_token.strip()
        if not token:
            continue
        if isinstance(current, list):
            current = current[int(token)]
            continue
        if not isinstance(current, Mapping):
            return None
        current = current.get(token)
        if current is None:
            return None
    return current


def _json_request(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    request = Request(
        url=url,
        data=body,
        headers=request_headers,
        method=method.upper(),
    )
    try:
        with _urlopen(request, timeout_seconds=timeout_seconds) as response:
            content = response.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - network failures
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method.upper()} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"{method.upper()} {url} failed: {exc.reason}") from exc

    if not content.strip():
        return {}
    return json.loads(content)


def _download_bytes(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float = 300.0,
) -> bytes:
    request = Request(url=url, headers=dict(headers), method="GET")
    try:
        with _urlopen(request, timeout_seconds=timeout_seconds) as response:
            return response.read()
    except HTTPError as exc:  # pragma: no cover - network failures
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


class HttpPollingProvider(ApiProvider):
    def __init__(self, *, provider_name: str, api_config: Mapping[str, Any]) -> None:
        self.provider_name = provider_name
        self.api_config = dict(api_config)
        self.base_url = self._resolve_base_url()
        self.default_timeout_seconds = float(self.api_config.get("request_timeout_seconds", 300))

    def _resolve_base_url(self) -> str | None:
        """Resolve base url -> str | None."""
        if self.api_config.get("base_url"):
            return str(self.api_config["base_url"]).rstrip("/")
        env_name = self.api_config.get("base_url_env")
        if env_name:
            env_value = os.environ.get(str(env_name))
            if env_value:
                return env_value.rstrip("/")
        return None

    def _headers(self) -> dict[str, str]:
        """Headers -> dict[str, str]."""
        headers = {
            str(key): str(value)
            for key, value in dict(self.api_config.get("headers", {})).items()
        }
        api_key_env = self.api_config.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(str(api_key_env))
            if not api_key:
                raise RuntimeError(f"required API credential env var is missing: {api_key_env}")
            auth_header = str(self.api_config.get("auth_header", "Authorization"))
            auth_scheme = str(self.api_config.get("auth_scheme", "Bearer")).strip()
            if auth_scheme:
                headers[auth_header] = f"{auth_scheme} {api_key}"
            else:
                headers[auth_header] = api_key
        return headers

    def _task_id(self, payload: Mapping[str, Any]) -> str:
        task_id_path = str(self.api_config.get("task_id_path", "task_id"))
        task_id = _extract_path(payload, task_id_path)
        if task_id is None or not str(task_id).strip():
            raise RuntimeError(
                f"task id was not found at path {task_id_path!r} in submit response: {payload}"
            )
        return str(task_id)

    def _status(self, payload: Mapping[str, Any]) -> str:
        status_path = str(self.api_config.get("status_path", "status"))
        status = _extract_path(payload, status_path)
        return "" if status is None else str(status)

    def _artifact_url(self, payload: Mapping[str, Any]) -> str | None:
        artifact_url_path = self.api_config.get("artifact_url_path")
        artifact_url = _extract_path(payload, None if artifact_url_path is None else str(artifact_url_path))
        return None if artifact_url is None else str(artifact_url)

    def submit(self, payload: dict[str, Any]) -> ApiJobHandle:
        submit_route = payload.get("submit_route") or self.api_config.get("submit_route")
        if submit_route is None:
            raise ValueError("submit_route is required for HTTP polling providers")
        request_body = dict(payload.get("request_body", {}))
        response = _json_request(
            method=str(payload.get("submit_method", "POST")),
            url=_resolve_url(base_url=self.base_url, route_or_url=str(submit_route)),
            headers=self._headers(),
            payload=request_body,
            timeout_seconds=_float_or_default(
                payload.get("request_timeout_seconds"),
                self.default_timeout_seconds,
            ),
        )
        return ApiJobHandle(
            provider=self.provider_name,
            job_id=self._task_id(response),
            raw=response,
        )

    def wait(self, handle: ApiJobHandle) -> dict[str, Any]:
        status_route = self.api_config.get("status_route")
        if status_route is None:
            return dict(handle.raw)

        success_statuses = {
            str(value).lower()
            for value in self.api_config.get(
                "success_statuses",
                ["success", "succeed", "succeeded", "completed", "done", "finished"],
            )
        }
        failure_statuses = {
            str(value).lower()
            for value in self.api_config.get(
                "failure_statuses",
                ["failed", "error", "canceled", "cancelled"],
            )
        }
        timeout_seconds = float(self.api_config.get("timeout_seconds", 900))
        poll_interval_seconds = float(self.api_config.get("poll_interval_seconds", 10))
        started = time.time()

        while True:
            response = _json_request(
                method="GET",
                url=_resolve_url(
                    base_url=self.base_url,
                    route_or_url=str(status_route).format(task_id=handle.job_id),
                ),
                headers=self._headers(),
                timeout_seconds=self.default_timeout_seconds,
            )
            status = self._status(response).lower()
            artifact_url = self._artifact_url(response)
            if artifact_url and not status:
                return response
            if status in failure_statuses:
                raise RuntimeError(
                    f"API job {handle.job_id} failed with status {status!r}: {response}"
                )
            if status in success_statuses:
                return response
            if time.time() - started >= timeout_seconds:
                raise TimeoutError(f"API job {handle.job_id} timed out after {timeout_seconds}s")
            time.sleep(poll_interval_seconds)

    def download(self, result: dict[str, Any], output_path: Path) -> ApiGenerationResult:
        artifact_url = self._artifact_url(result)
        if not artifact_url:
            raise RuntimeError(f"artifact URL is missing from API result: {result}")

        download_headers = {
            str(key): str(value)
            for key, value in dict(self.api_config.get("download_headers", {})).items()
        }
        if not download_headers and bool(self.api_config.get("reuse_auth_for_download", False)):
            download_headers = self._headers()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            _download_bytes(
                url=artifact_url,
                headers=download_headers,
                timeout_seconds=_float_or_default(
                    self.api_config.get("download_timeout_seconds"),
                    300,
                ),
            )
        )
        return ApiGenerationResult(
            provider=self.provider_name,
            artifact_path=output_path,
            response=dict(result),
            metadata={
                "artifact_url": artifact_url,
                "status": self._status(result),
            },
        )


__all__ = ["HttpPollingProvider"]
