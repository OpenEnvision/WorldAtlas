from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from dataclasses import asdict, dataclass
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Sequence


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".gif", ".png", ".jpg", ".jpeg", ".webp"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
SKIP_DIR_NAMES = {
    ".cache",
    ".git",
    "__pycache__",
    "_conditioning",
    "media",
    "showcase",
}
CAMERA_TOKENS = {
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_back",
    "cam_back_left",
    "cam_back_right",
    "front",
    "front_left",
    "front_right",
    "back",
    "back_left",
    "back_right",
    "left",
    "right",
}
VALID_MODES = {"multi_view", "method_compare", "gif_compare"}


@dataclass(frozen=True)
class ShowcaseMedia:
    source_path: str
    method: str
    sample_id: str
    suite: str | None
    view: str | None
    role: str
    media_type: str
    prompt: str | None = None
    status: str | None = None
    error: str | None = None
    url: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix == ".gif":
        return "gif"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def _is_media_path(path: Path, extensions: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _generation_record_paths(root: Path) -> list[Path]:
    return sorted(root.glob("generation_records*.jsonl"))


def _resolve_record_path(root: Path, raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _sample_view_from_relative_path(relative_path: Path) -> tuple[str, str | None]:
    stem = relative_path.stem
    lowered_stem = stem.lower()
    if relative_path.parent != Path(".") and lowered_stem in CAMERA_TOKENS:
        return relative_path.parent.as_posix(), stem

    for token in sorted(CAMERA_TOKENS, key=len, reverse=True):
        suffix = f"_{token}"
        if lowered_stem.endswith(suffix):
            return stem[: -len(suffix)], stem[-len(token) :]

    if relative_path.parent != Path("."):
        parent_name = relative_path.parent.as_posix()
        parts = relative_path.parts
        if len(parts) >= 2 and parts[-2].lower() in CAMERA_TOKENS:
            return "/".join(parts[:-2] + (stem,)), parts[-2]
        return parent_name, None
    return stem, None


def _iter_media_files(root: Path, extensions: set[str]) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name not in SKIP_DIR_NAMES
        ]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            if _is_media_path(path, extensions):
                paths.append(path.resolve())
    return sorted(paths)


def _discover_from_generation_records(
    method: str,
    root: Path,
    extensions: set[str],
    *,
    include_conditioning: bool,
) -> list[ShowcaseMedia]:
    entries: list[ShowcaseMedia] = []
    for records_path in _generation_record_paths(root):
        for record in _read_jsonl(records_path):
            prediction_path = _resolve_record_path(root, record.get("prediction_path"))
            if prediction_path and _is_media_path(prediction_path, extensions):
                entries.append(
                    ShowcaseMedia(
                        source_path=str(prediction_path),
                        method=method,
                        sample_id=str(record.get("sample_id") or prediction_path.stem),
                        suite=str(record["suite"]) if record.get("suite") else None,
                        view=str(record["view"]) if record.get("view") else None,
                        role="prediction",
                        media_type=_media_type(prediction_path),
                        prompt=str(record["prompt"]) if record.get("prompt") else None,
                        status=str(record["status"]) if record.get("status") else None,
                        error=str(record["error"]) if record.get("error") else None,
                    )
                )
            if include_conditioning:
                conditioning_path = _resolve_record_path(
                    root,
                    record.get("conditioning_input") or record.get("conditioning_image"),
                )
                if conditioning_path and _is_media_path(conditioning_path, extensions):
                    entries.append(
                        ShowcaseMedia(
                            source_path=str(conditioning_path),
                            method=f"{method}:conditioning",
                            sample_id=str(record.get("sample_id") or conditioning_path.stem),
                            suite=str(record["suite"]) if record.get("suite") else None,
                            view=str(record["view"]) if record.get("view") else None,
                            role="conditioning",
                            media_type=_media_type(conditioning_path),
                            prompt=str(record["prompt"]) if record.get("prompt") else None,
                        )
                    )
    return entries


def _discover_from_media_tree(
    method: str,
    root: Path,
    extensions: set[str],
) -> list[ShowcaseMedia]:
    entries: list[ShowcaseMedia] = []
    for path in _iter_media_files(root, extensions):
        relative_path = path.relative_to(root)
        sample_id, view = _sample_view_from_relative_path(relative_path)
        entries.append(
            ShowcaseMedia(
                source_path=str(path),
                method=method,
                sample_id=sample_id,
                suite=relative_path.parts[0] if len(relative_path.parts) > 1 else None,
                view=view,
                role="prediction",
                media_type=_media_type(path),
            )
        )
    return entries


def _load_benchmark_references(
    manifest_path: Path,
    extensions: set[str],
) -> list[ShowcaseMedia]:
    entries: list[ShowcaseMedia] = []
    for record in _read_jsonl(manifest_path):
        sample_id = str(record.get("sample_id") or record.get("id") or "")
        if not sample_id:
            continue
        reference_path = _resolve_record_path(manifest_path.parent, record.get("reference_path"))
        if not reference_path or not _is_media_path(reference_path, extensions):
            continue
        entries.append(
            ShowcaseMedia(
                source_path=str(reference_path),
                method="reference",
                sample_id=sample_id,
                suite=str(record["suite"]) if record.get("suite") else None,
                view=None,
                role="reference",
                media_type=_media_type(reference_path),
                prompt=str(record["prompt"]) if record.get("prompt") else None,
            )
        )
    return entries


def _filter_mode(entries: list[ShowcaseMedia], mode: str) -> list[ShowcaseMedia]:
    if mode == "gif_compare":
        return [entry for entry in entries if entry.media_type in {"gif", "image"}]
    if mode in {"method_compare", "multi_view"}:
        return [entry for entry in entries if entry.media_type in {"video", "gif", "image"}]
    raise ValueError(f"unsupported showcase mode: {mode}")


def discover_showcase_media(
    method_roots: dict[str, Path],
    *,
    mode: str = "method_compare",
    extensions: set[str] | None = None,
    benchmark_manifest: Path | None = None,
    include_reference: bool = False,
    include_conditioning: bool = False,
) -> list[ShowcaseMedia]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    selected_extensions = extensions or MEDIA_EXTENSIONS
    entries: list[ShowcaseMedia] = []
    for method, root in method_roots.items():
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        record_entries = _discover_from_generation_records(
            method,
            root,
            selected_extensions,
            include_conditioning=include_conditioning,
        )
        entries.extend(record_entries or _discover_from_media_tree(method, root, selected_extensions))

    if include_reference:
        if benchmark_manifest is None:
            raise ValueError("--include-reference requires --benchmark-manifest")
        entries.extend(_load_benchmark_references(benchmark_manifest.expanduser().resolve(), selected_extensions))

    filtered = _filter_mode(entries, mode)
    return sorted(filtered, key=lambda item: (item.sample_id, item.method, item.view or "", item.source_path))


def _safe_slug(value: str, *, fallback: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:96] or fallback


def _media_link_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    stem = _safe_slug(path.stem, fallback="media")
    return f"{digest}-{stem}{path.suffix.lower()}"


def _link_media_file(source_path: Path, output_dir: Path) -> str:
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    destination = media_dir / _media_link_name(source_path)
    if destination.exists() or destination.is_symlink():
        try:
            if destination.resolve() == source_path.resolve():
                return destination.relative_to(output_dir).as_posix()
        except FileNotFoundError:
            pass
        destination.unlink()
    try:
        relative_target = os.path.relpath(source_path, start=destination.parent)
        destination.symlink_to(relative_target)
    except OSError:
        shutil.copy2(source_path, destination)
    return destination.relative_to(output_dir).as_posix()


def _sample_limit(entries: list[ShowcaseMedia], limit: int | None) -> list[ShowcaseMedia]:
    if limit is None:
        return entries
    allowed: set[str] = set()
    for entry in entries:
        allowed.add(entry.sample_id)
        if len(allowed) >= limit:
            break
    return [entry for entry in entries if entry.sample_id in allowed]


def _manifest_payload(
    *,
    title: str,
    mode: str,
    entries: list[ShowcaseMedia],
) -> dict[str, Any]:
    samples = sorted({entry.sample_id for entry in entries})
    methods = sorted({entry.method for entry in entries})
    suites = sorted({entry.suite for entry in entries if entry.suite})
    views = sorted({entry.view for entry in entries if entry.view})
    return {
        "title": title,
        "mode": mode,
        "samples": samples,
        "methods": methods,
        "suites": suites,
        "views": views,
        "items": [entry.to_payload() for entry in entries],
        "summary": {
            "samples": len(samples),
            "methods": len(methods),
            "suites": len(suites),
            "views": len(views),
            "media": len(entries),
        },
    }


def _render_html(title: str, payload: dict[str, Any]) -> str:
    escaped_title = html.escape(title)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #65758b;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-dark: #115e59;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
    }}
    .header-inner {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: end;
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 20px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f9fafb;
      padding: 4px 8px;
      white-space: nowrap;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(180px, 260px) minmax(120px, 180px) minmax(120px, 180px);
      gap: 10px;
    }}
    input, select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      padding: 7px 9px;
      font: inherit;
      font-size: 14px;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 32px;
    }}
    .empty {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      padding: 24px;
    }}
    .sample-section {{
      margin: 0 0 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .sample-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .sample-title {{
      min-width: 0;
      font-size: 15px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .sample-meta {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      padding: 12px;
    }}
    .media-card {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }}
    .media-title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
    }}
    .media-role {{
      color: var(--muted);
      font-weight: 500;
    }}
    video, img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #111827;
    }}
    .media-path {{
      padding: 8px 10px 10px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .prompt {{
      padding: 0 10px 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    @media (max-width: 820px) {{
      .header-inner {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
      .sample-header {{ align-items: flex-start; flex-direction: column; }}
      .sample-meta {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>{escaped_title}</h1>
        <div class="summary" id="summary"></div>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="Search sample, method, suite, prompt">
        <select id="suite-filter" aria-label="Suite filter"></select>
        <select id="view-filter" aria-label="View filter"></select>
      </div>
    </div>
  </header>
  <main id="app"></main>
  <script id="manifest-data" type="application/json">{payload_json}</script>
  <script>
    const manifest = JSON.parse(document.getElementById('manifest-data').textContent);
    const app = document.getElementById('app');
    const search = document.getElementById('search');
    const suiteFilter = document.getElementById('suite-filter');
    const viewFilter = document.getElementById('view-filter');

    function option(value, label) {{
      const el = document.createElement('option');
      el.value = value;
      el.textContent = label;
      return el;
    }}

    function fillSelect(select, values, allLabel) {{
      select.appendChild(option('', allLabel));
      values.forEach(value => select.appendChild(option(value, value)));
    }}

    function mediaNode(item) {{
      const node = item.media_type === 'video' ? document.createElement('video') : document.createElement('img');
      node.src = item.url;
      if (item.media_type === 'video') {{
        node.controls = true;
        node.loop = true;
        node.muted = true;
        node.preload = 'metadata';
      }} else {{
        node.loading = 'lazy';
        node.alt = `${{item.method}} ${{item.sample_id}}`;
      }}
      return node;
    }}

    function renderMedia(item) {{
      const card = document.createElement('article');
      card.className = 'media-card';
      const title = document.createElement('div');
      title.className = 'media-title';
      const left = document.createElement('span');
      left.textContent = item.view ? `${{item.method}} · ${{item.view}}` : item.method;
      const right = document.createElement('span');
      right.className = 'media-role';
      right.textContent = item.role;
      title.append(left, right);
      card.append(title, mediaNode(item));
      const path = document.createElement('div');
      path.className = 'media-path';
      path.textContent = item.source_path;
      card.appendChild(path);
      if (item.prompt) {{
        const prompt = document.createElement('div');
        prompt.className = 'prompt';
        prompt.textContent = item.prompt;
        card.appendChild(prompt);
      }}
      return card;
    }}

    function groupItems(items) {{
      const groups = new Map();
      items.forEach(item => {{
        const key = item.sample_id;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
      }});
      return Array.from(groups.entries()).sort((a, b) => a[0].localeCompare(b[0]));
    }}

    function render() {{
      const query = search.value.trim().toLowerCase();
      const suite = suiteFilter.value;
      const view = viewFilter.value;
      const items = manifest.items.filter(item => {{
        if (suite && item.suite !== suite) return false;
        if (view && item.view !== view) return false;
        if (!query) return true;
        return [item.sample_id, item.method, item.suite, item.view, item.prompt, item.role]
          .filter(Boolean)
          .some(value => String(value).toLowerCase().includes(query));
      }});
      app.replaceChildren();
      if (!items.length) {{
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'No media matches the current filters.';
        app.appendChild(empty);
        return;
      }}
      groupItems(items).forEach(([sampleId, group]) => {{
        const section = document.createElement('section');
        section.className = 'sample-section';
        const header = document.createElement('div');
        header.className = 'sample-header';
        const title = document.createElement('div');
        title.className = 'sample-title';
        title.textContent = sampleId;
        const meta = document.createElement('div');
        meta.className = 'sample-meta';
        const methods = [...new Set(group.map(item => item.method))].sort();
        const suites = [...new Set(group.map(item => item.suite).filter(Boolean))].sort();
        meta.textContent = `${{group.length}} media · ${{methods.join(', ')}}${{suites.length ? ' · ' + suites.join(', ') : ''}}`;
        header.append(title, meta);
        const grid = document.createElement('div');
        grid.className = 'media-grid';
        group
          .slice()
          .sort((a, b) => [a.method, a.view || '', a.role].join('|').localeCompare([b.method, b.view || '', b.role].join('|')))
          .forEach(item => grid.appendChild(renderMedia(item)));
        section.append(header, grid);
        app.appendChild(section);
      }});
    }}

    document.getElementById('summary').replaceChildren(
      ...Object.entries(manifest.summary).map(([key, value]) => {{
        const node = document.createElement('span');
        node.textContent = `${{value}} ${{key}}`;
        return node;
      }})
    );
    fillSelect(suiteFilter, manifest.suites, 'All suites');
    fillSelect(viewFilter, manifest.views, 'All views');
    [search, suiteFilter, viewFilter].forEach(node => node.addEventListener('input', render));
    render();
  </script>
</body>
</html>
"""


def build_showcase(
    method_roots: dict[str, Path],
    *,
    output_dir: Path,
    mode: str = "method_compare",
    title: str | None = None,
    benchmark_manifest: Path | None = None,
    include_reference: bool = False,
    include_conditioning: bool = False,
    extensions: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_showcase_media(
        method_roots,
        mode=mode,
        extensions=extensions,
        benchmark_manifest=benchmark_manifest,
        include_reference=include_reference,
        include_conditioning=include_conditioning,
    )
    entries = _sample_limit(entries, limit)
    linked_entries = [
        ShowcaseMedia(
            source_path=entry.source_path,
            method=entry.method,
            sample_id=entry.sample_id,
            suite=entry.suite,
            view=entry.view,
            role=entry.role,
            media_type=entry.media_type,
            prompt=entry.prompt,
            status=entry.status,
            error=entry.error,
            url=_link_media_file(Path(entry.source_path), output_dir),
        )
        for entry in entries
    ]
    display_title = title or f"WorldArena {mode.replace('_', ' ').title()}"
    payload = _manifest_payload(title=display_title, mode=mode, entries=linked_entries)
    manifest_path = output_dir / "showcase_manifest.json"
    html_path = output_dir / "index.html"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(_render_html(display_title, payload), encoding="utf-8")
    return {
        "mode": mode,
        "title": display_title,
        "output_dir": str(output_dir),
        "html_path": str(html_path),
        "manifest_path": str(manifest_path),
        "summary": payload["summary"],
    }


def _parse_method_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"method spec must be NAME=PATH: {spec!r}")
        return name.strip(), Path(raw_path.strip())
    path = Path(spec)
    return path.name, path


def _method_roots_from_args(args: argparse.Namespace) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for spec in args.method or []:
        name, path = _parse_method_spec(spec)
        roots[name] = path
    for raw_root in args.predictions_root or []:
        path = Path(raw_root)
        roots[path.name] = path
    if roots:
        return roots

    generated_results = Path(args.generated_results)
    if not generated_results.exists():
        raise FileNotFoundError(
            f"no method roots supplied and default generated results root does not exist: {generated_results}"
        )
    children = sorted(path for path in generated_results.iterdir() if path.is_dir())
    if children:
        return {path.name: path for path in children}
    return {generated_results.name: generated_results}


def _extensions_from_args(raw_extensions: str | None) -> set[str] | None:
    if not raw_extensions:
        return None
    extensions = {
        item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}"
        for item in raw_extensions.split(",")
        if item.strip()
    }
    unsupported = extensions - MEDIA_EXTENSIONS
    if unsupported:
        raise ValueError(f"unsupported media extensions: {sorted(unsupported)}")
    return extensions


def add_showcase_arguments(parser: argparse.ArgumentParser, *, default_mode: str = "method_compare") -> None:
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default=default_mode)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method root as NAME=PATH. Can be passed multiple times.",
    )
    parser.add_argument(
        "--predictions-root",
        action="append",
        default=[],
        help="Prediction root; the directory name is used as the method name.",
    )
    parser.add_argument(
        "--generated-results",
        default="generated_results",
        help="WorldLens-style parent directory whose children are method roots.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--benchmark-manifest", default=None)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--include-conditioning", action="store_true")
    parser.add_argument("--extensions", default=None, help="Comma-separated media extensions to include.")
    parser.add_argument("--limit", type=int, default=None, help="Limit rendered sample count.")
    parser.add_argument("--serve", action="store_true", help="Serve the generated static viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)


def serve_showcase(output_dir: Path, *, host: str, port: int) -> None:
    output_dir = output_dir.expanduser().resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(output_dir), **kwargs)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving WorldArena showcase at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def run_showcase_from_args(args: argparse.Namespace) -> dict[str, Any]:
    method_roots = _method_roots_from_args(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path(".cache/showcase") / args.mode
    result = build_showcase(
        method_roots,
        output_dir=output_dir,
        mode=args.mode,
        title=args.title,
        benchmark_manifest=Path(args.benchmark_manifest) if args.benchmark_manifest else None,
        include_reference=args.include_reference,
        include_conditioning=args.include_conditioning,
        extensions=_extensions_from_args(args.extensions),
        limit=args.limit,
    )
    if args.serve:
        serve_showcase(Path(result["output_dir"]), host=args.host, port=args.port)
    return result


def main(argv: Sequence[str] | None = None, *, default_mode: str = "method_compare") -> int:
    parser = argparse.ArgumentParser(description="Build a static WorldArena media showcase viewer.")
    add_showcase_arguments(parser, default_mode=default_mode)
    args = parser.parse_args(argv)
    result = run_showcase_from_args(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main_with_mode(mode: str) -> int:
    return main(default_mode=mode)


if __name__ == "__main__":
    raise SystemExit(main())
