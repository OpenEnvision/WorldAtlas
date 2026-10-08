#!/usr/bin/env python3
"""Fetch selected upstream revisions and apply the recorded runtime patches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(path), *args], check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def prepare(root: Path, name: str, source: dict, *, check_only: bool) -> None:
    relative = Path(source["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "thirdparty":
        raise ValueError(f"Invalid dependency path: {relative}")
    revision = source["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{name}: expected a full Git commit ID")
    destination = root / relative
    # An empty submodule directory can inherit the parent repository's Git state.
    probe = git(destination, "rev-parse", "--show-toplevel", check=False) if destination.is_dir() else None
    is_checkout = probe is not None and probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == destination.resolve()
    if not is_checkout:
        if check_only:
            raise RuntimeError(f"{name}: missing checkout; run this command without --check")
        if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
            raise RuntimeError(f"{name}: {destination} contains files; choose an empty checkout directory")
        destination.mkdir(parents=True, exist_ok=True)
        git(destination, "init")
        git(destination, "remote", "add", "origin", source["url"])
        git(destination, "fetch", "--depth", "1", "origin", revision)
        git(destination, "checkout", "--detach", "FETCH_HEAD")
    actual = git(destination, "rev-parse", "HEAD").stdout.strip()
    if actual != revision:
        raise RuntimeError(
            f"{name}: expected {revision}, found {actual}. Preserve local changes and "
            "use a fresh directory, or explicitly check out the pinned revision."
        )
    if not check_only:
        git(destination, "submodule", "update", "--init", "--recursive")
    for patch in source.get("patches", []):
        patch_path = (root / patch["file"]).resolve()
        if not patch_path.is_relative_to(root.resolve()):
            raise ValueError(f"{name}: patch must be inside the release tree")
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != patch["sha256"]:
            raise ValueError(f"{name}: patch checksum mismatch: {patch_path.name}")
        if git(destination, "apply", "--reverse", "--check", str(patch_path), check=False).returncode == 0:
            continue
        if check_only:
            raise RuntimeError(f"{name}: runtime patch is missing: {patch_path.name}")
        git(destination, "apply", "--check", str(patch_path))
        git(destination, "apply", str(patch_path))
    print(f"{name}: {revision} (source and patches ready)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dependency names from --list")
    parser.add_argument("--list", action="store_true", help="List available pinned dependencies")
    parser.add_argument("--all", action="store_true", help="Prepare every dependency; downloads can be large")
    parser.add_argument("--check", action="store_true", help="Verify revisions and patches without changing files")
    args = parser.parse_args()
    sources = json.loads((ROOT / "thirdparty/sources.lock.json").read_text())["sources"]
    if args.list:
        for name, source in sources.items():
            print(f"{name:40} {source['revision']}  {source['url']}")
        return 0
    names = list(sources) if args.all else list(dict.fromkeys(args.names))
    if not names:
        parser.error("select one or more names, --all, or --list")
    unknown = set(names) - sources.keys()
    if unknown:
        parser.error(f"unknown dependencies: {', '.join(sorted(unknown))}; see --list")
    for name in names:
        try:
            prepare(ROOT, name, sources[name], check_only=args.check)
        except subprocess.CalledProcessError as exc:
            print(f"{name}: {exc.stderr or str(exc)}", file=sys.stderr)
            return 1
        except (OSError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
