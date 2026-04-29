from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


LOG = logging.getLogger("rehome_vipe_annotations")
DEFAULT_SOURCE_ROOT = Path("/share/project/yesliu/hjd/data/videos_test_split_4971")
DEFAULT_TARGET_ROOT = Path("/videos_test_split_4971")
BACKUP_DIRNAME = ".annotations_vipe.rehome_backup"


@dataclass(frozen=True, slots=True)
class MigrationJob:
    index: int
    total: int
    source_ann: Path
    target_ann: Path
    size_kib: int

    @property
    def label(self) -> str:
        return str(self.source_ann)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move source-side annotations_vipe directories onto a root-backed "
            "mirror and replace the source directories with symlinks."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def dir_size_kib(path: Path) -> int:
    result = run_command(["du", "-sk", str(path)])
    return int(result.stdout.split()[0])


def discover_jobs(source_root: Path, target_root: Path) -> list[MigrationJob]:
    source_ann_dirs = sorted(path for path in source_root.rglob("annotations_vipe") if path.is_dir() and not path.is_symlink())
    sized: list[tuple[int, Path]] = []
    for path in source_ann_dirs:
        sized.append((dir_size_kib(path), path))
    sized.sort(reverse=True, key=lambda item: item[0])
    jobs: list[MigrationJob] = []
    total = len(sized)
    for index, (size_kib, source_ann) in enumerate(sized, start=1):
        relative = source_ann.relative_to(source_root)
        target_ann = target_root / relative
        jobs.append(
            MigrationJob(
                index=index,
                total=total,
                source_ann=source_ann,
                target_ann=target_ann,
                size_kib=size_kib,
            )
        )
    return jobs


def ensure_synced(source_dir: Path, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    run_command(["rsync", "-a", "--delete", f"{source_dir}/", f"{target_dir}/"])


def replace_with_symlink(source_ann: Path, target_ann: Path, backup_ann: Path) -> None:
    if source_ann.is_symlink():
        if source_ann.resolve() != target_ann.resolve():
            raise RuntimeError(f"{source_ann} already points to {source_ann.resolve()}, expected {target_ann}")
        return
    if source_ann.exists():
        source_ann.rename(backup_ann)
    source_ann.symlink_to(target_ann)
    run_command(["rm", "-rf", str(backup_ann)])


def migrate_one(job: MigrationJob) -> str:
    source_ann = job.source_ann
    target_ann = job.target_ann
    backup_ann = source_ann.parent / BACKUP_DIRNAME
    if source_ann.is_symlink():
        if source_ann.resolve() != target_ann.resolve():
            raise RuntimeError(f"{source_ann} already points to {source_ann.resolve()}, expected {target_ann}")
        LOG.info("[%d/%d] already linked %s", job.index, job.total, source_ann)
        return "already_linked"
    if backup_ann.exists():
        if source_ann.exists():
            raise RuntimeError(f"both source and backup exist for {source_ann}")
        working_source = backup_ann
    else:
        working_source = source_ann
    gib = job.size_kib / 1024.0 / 1024.0
    LOG.info("[%d/%d] syncing %.2f GiB %s -> %s", job.index, job.total, gib, working_source, target_ann)
    ensure_synced(working_source, target_ann)
    replace_with_symlink(source_ann, target_ann, backup_ann)
    LOG.info("[%d/%d] linked %s -> %s", job.index, job.total, source_ann, target_ann)
    return "linked"


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(source_root, target_root)
    if not jobs:
        LOG.info("no source-side annotations_vipe directories found under %s", source_root)
        return 0
    LOG.info("discovered %d annotations_vipe directories to rehome", len(jobs))
    failures: list[tuple[MigrationJob, BaseException]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map: dict[Future[str], MigrationJob] = {executor.submit(migrate_one, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                future.result()
            except BaseException as exc:  # pragma: no cover - operational script
                failures.append((job, exc))
                LOG.error("failed to migrate %s: %s", job.source_ann, exc)
    if failures:
        LOG.error("%d migration jobs failed", len(failures))
        for job, exc in failures:
            LOG.error("job failed: %s (%s)", job.source_ann, exc)
        return 1
    LOG.info("completed all %d migration jobs", len(jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
