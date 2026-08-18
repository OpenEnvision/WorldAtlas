"""Export documentation content into static site JavaScript bundles."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


def _source_to_route(source_path: str) -> str:
    if source_path == "index.md":
        return "/doc/"

    stem = source_path[:-3] if source_path.endswith(".md") else source_path
    if stem.endswith("/index"):
        stem = stem[: -len("/index")]
    elif stem == "index":
        return "/doc/"

    segments = [part.replace("_", "-") for part in stem.split("/") if part]
    return f"/doc/{'/'.join(segments)}/"


def _extract_title(content: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+?)(?:\s+\{#[^}]+\})?\s*$", content, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def _strip_markdown(value: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(^|\s)#{1,6}\s*", r"\1", text)
    text = re.sub(r"[*_>#~]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_summary(content: str) -> str:
    return _strip_markdown(_extract_hero_markdown(content))


def _extract_hero_markdown(content: str) -> str:
    return _extract_raw_hero_markdown(content)


def _strip_metric_examples_block(content: str) -> str:
    """Remove injected metric-example galleries from markdown before hero extraction."""
    return re.sub(
        r"<!--\s*metric-examples:start\s*-->[\s\S]*?<!--\s*metric-examples:end\s*-->",
        "",
        content,
        flags=re.IGNORECASE,
    )


def _is_reference_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^##\s+references\b", stripped, flags=re.IGNORECASE)) or "{#reference}" in stripped


def _find_doc_hero_bounds(content: str) -> tuple[list[str], int, int]:
    lines = _strip_metric_examples_block(content).replace("\r\n", "\n").split("\n")
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            index += 1
            break
        index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    hero_start = index
    seen_h2 = False

    while index < len(lines):
        stripped = lines[index].strip()
        if _is_reference_heading(stripped):
            break
        if stripped.startswith("## "):
            if seen_h2:
                break
            seen_h2 = True
        index += 1

    return lines, hero_start, index


def _extract_raw_hero_markdown(content: str) -> str:
    lines, hero_start, hero_end = _find_doc_hero_bounds(content)
    return "\n".join(lines[hero_start:hero_end]).strip()


def _derive_tags(section: str, source_path: str) -> list[str]:
    tags = [section]
    lowered = source_path.lower()
    for token in ("benchmark", "metrics", "physics", "dataset", "runtime", "development"):
        if token in lowered:
            tags.append(token)
    if "dimensions" in lowered:
        tags.extend(["metrics", "dimensions"])
    if "camera" in lowered:
        tags.append("camera")
    if "depth" in lowered:
        tags.append("depth")
    if "long_sequence" in lowered or "memory" in lowered:
        tags.extend(["long sequence", "memory"])
    if "action_control" in lowered:
        tags.append("action control")
    if "consistency_3d_4d" in lowered or "3d" in lowered or "4d" in lowered:
        tags.extend(["3D", "4D"])
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tag)
    return deduped


def _adapt_content_for_site(content: str) -> str:
    replacements = (
        ("../../../assets/metric_examples/", "/assets/worldarena/assets/metric_examples/"),
        ("../../assets/metric_examples/", "/assets/worldarena/assets/metric_examples/"),
        ("../../../assets/img/", "/assets/worldarena/assets/img/"),
        ("../../assets/img/", "/assets/worldarena/assets/img/"),
        ("../../generated/", "/assets/worldarena/generated/"),
        ("../generated/", "/assets/worldarena/generated/"),
    )
    for source, target in replacements:
        content = content.replace(source, target)
    return content


def _sync_tree_pruned(source_dir: Path, target_dir: Path) -> None:
    """Copy a docs asset tree and remove stale regular files from previous exports."""
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    source_files = {
        path.relative_to(source_dir)
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    for target_file in sorted(target_dir.rglob("*")):
        if not target_file.is_file():
            continue
        if target_file.name.startswith(".nfs"):
            continue
        if target_file.relative_to(target_dir) not in source_files:
            target_file.unlink()


def _ensure_asset_symlink(source_dir: Path, target_dir: Path) -> None:
    if target_dir.is_symlink():
        if target_dir.resolve() == source_dir.resolve():
            return
        target_dir.unlink()
    elif target_dir.exists():
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.symlink_to(source_dir, target_is_directory=True)


def _sync_docs_assets(root: Path, site: Path) -> None:
    source_root = root / "docs" / "assets"
    target_root = site / "assets" / "worldarena" / "assets"
    for asset_dir in ("img", "metric_examples"):
        source_dir = source_root / asset_dir
        if not source_dir.is_dir():
            continue

        target_dir = target_root / asset_dir
        _ensure_asset_symlink(source_dir, target_dir)

    generated_source = root / "docs" / "generated"
    generated_target = site / "assets" / "worldarena" / "generated"
    if generated_source.is_dir():
        _ensure_asset_symlink(generated_source, generated_target)


def _nav_link(title: str, source_path: str) -> dict[str, Any]:
    return {
        "title": title,
        "route": _source_to_route(source_path),
        "sourcePath": source_path,
    }


def _parse_nav_links(entries: list[Any]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for title, value in entry.items():
            if isinstance(value, str):
                links.append(_nav_link(title, value))
            elif isinstance(value, list):
                links.append({"title": title, "children": _parse_nav_links(value)})
    return links


def _parse_nav_groups(nav: list[Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for entry in nav:
        if not isinstance(entry, dict):
            continue
        for title, value in entry.items():
            if isinstance(value, str):
                groups.append({"title": title, "links": [_nav_link(title, value)]})
            elif isinstance(value, list):
                groups.append({"title": title, "links": _parse_nav_links(value)})
    return groups


def _load_site_nav_config(project_root: Path) -> dict[str, Any]:
    nav_path = project_root / "docs" / "site-nav.yaml"
    payload = yaml.safe_load(nav_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("docs/site-nav.yaml must be a mapping")
    return payload


def _load_i18n_nav_translations(project_root: Path) -> dict[str, str]:
    translations = _load_site_nav_config(project_root).get("nav_translations", {})
    return translations if isinstance(translations, dict) else {}


def _localize_navigation(navigation_en: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    nav_translations = _load_i18n_nav_translations(project_root)

    def localize_node(node: dict[str, Any]) -> dict[str, Any]:
        localized = dict(node)
        localized["title"] = nav_translations.get(node["title"], node["title"])
        if "children" in node:
            localized["children"] = [localize_node(child) for child in node["children"]]
        return localized

    localized_groups: list[dict[str, Any]] = []
    for group in navigation_en:
        localized_groups.append(
            {
                "title": nav_translations.get(group["title"], group["title"]),
                "links": [localize_node(link) for link in group["links"]],
            }
        )
    return localized_groups


def _walk_link(link: dict[str, Any], section: str, collected: list[tuple[str, str]]) -> None:
    if "sourcePath" in link:
        collected.append((section, link["sourcePath"]))
        return
    for child in link.get("children", []):
        _walk_link(child, section, collected)


def _load_site_nav(project_root: Path) -> list[Any]:
    nav = _load_site_nav_config(project_root).get("nav")
    if not isinstance(nav, list):
        raise ValueError("docs/site-nav.yaml nav must be a list")
    return nav


def _load_hidden_pages(project_root: Path) -> list[str]:
    hidden = _load_site_nav_config(project_root).get("hidden_pages", [])
    if hidden is None:
        return []
    if not isinstance(hidden, list):
        raise ValueError("docs/site-nav.yaml hidden_pages must be a list")
    return [str(path) for path in hidden if isinstance(path, str)]


def _read_locale_page(docs_root: Path, locale: str, source_path: str) -> str:
    page_path = docs_root / locale / source_path
    if not page_path.exists():
        raise FileNotFoundError(f"Missing docs page: {page_path}")
    return page_path.read_text(encoding="utf-8")


def _extract_awesome_page(existing_content_path: Path) -> str:
    if not existing_content_path.exists():
        return 'export const awesomeWorldModelsPage = null;\n'
    text = existing_content_path.read_text(encoding="utf-8")
    marker = "export const awesomeWorldModelsPage"
    index = text.find(marker)
    if index < 0:
        return 'export const awesomeWorldModelsPage = null;\n'
    return text[index:].rstrip() + "\n"


def _render_js_module(
    *,
    pages_export_name: str,
    navigation_export_name: str,
    pages: list[dict[str, Any]],
    navigation: list[dict[str, Any]],
    awesome_page_js: str,
) -> str:
    pages_json = json.dumps(pages, ensure_ascii=False, indent=2)
    navigation_json = json.dumps(navigation, ensure_ascii=False, indent=2)
    return (
        f"export const {pages_export_name} = {pages_json};\n\n"
        f"export const {navigation_export_name} = {navigation_json};\n\n"
        f"{awesome_page_js}"
    )


def build_site_content(
    project_root: Path | None = None,
    site_root: Path | None = None,
) -> dict[str, Any]:
    """Export docs content into static site JavaScript bundles."""
    root = (project_root or Path.cwd()).resolve()
    site = (site_root or root / "site").resolve()
    docs_root = root / "docs"
    _sync_docs_assets(root, site)
    nav = _load_site_nav(root)
    navigation_en = _parse_nav_groups(nav)
    hidden_pages = _load_hidden_pages(root)

    collected: list[tuple[str, str]] = []
    for group in navigation_en:
        section = group["title"]
        for link in group["links"]:
            _walk_link(link, section, collected)

    seen_paths = {source_path for _, source_path in collected}
    for source_path in hidden_pages:
        if source_path in seen_paths:
            continue
        collected.append(("Benchmark", source_path))
        seen_paths.add(source_path)

    existing_content = site / "js" / "content.js"
    awesome_page_js = _extract_awesome_page(existing_content)

    locale_specs = [
        ("en", "docsPages", "docsNavigation", site / "js" / "content.js"),
        ("zh", "docsPagesZh", "docsNavigationZh", site / "js" / "content.zh.js"),
    ]

    result: dict[str, Any] = {"pages": {}, "output_files": []}
    for locale, pages_name, nav_name, output_path in locale_specs:
        pages: list[dict[str, Any]] = []
        for index, (section, source_path) in enumerate(collected, start=1):
            content = _adapt_content_for_site(_read_locale_page(docs_root, locale, source_path))
            nav_title = next(
                (
                    link.get("title")
                    for group in _parse_nav_groups(nav)
                    for link in _flatten_links(group["links"])
                    if link.get("sourcePath") == source_path
                ),
                section,
            )
            title = _extract_title(content, nav_title)
            pages.append(
                {
                    "id": f"worldarena-{locale}-{index:03d}",
                    "title": title,
                    "section": section,
                    "sourcePath": source_path,
                    "route": _source_to_route(source_path),
                    "summary": _extract_summary(content),
                    "tags": _derive_tags(section, source_path),
                    "content": content,
                }
            )

        navigation = navigation_en if locale == "en" else _localize_navigation(navigation_en, root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _render_js_module(
                pages_export_name=pages_name,
                navigation_export_name=nav_name,
                pages=pages,
                navigation=navigation,
                awesome_page_js=awesome_page_js if locale == "en" else "export const awesomeWorldModelsPage = null;\n",
            ),
            encoding="utf-8",
        )
        result["pages"][locale] = len(pages)
        result["output_files"].append(str(output_path))

    return result


def _flatten_links(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for link in links:
        if "sourcePath" in link:
            flattened.append(link)
        flattened.extend(_flatten_links(link.get("children", [])))
    return flattened
