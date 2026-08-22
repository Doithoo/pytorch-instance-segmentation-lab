"""Build or verify the deterministic self-contained Kaggle runner."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

from kaggle_runner import render_runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "docs" / "recorded-run" / "kaggle" / "run_kaggle.py"
ALLOWLIST = (
    "pyproject.toml",
    "configs/reference_maskrcnn.yaml",
    "data/manifests",
    "src/instance_segmenter",
    "scripts/download_data.py",
    "scripts/preview_dataset.py",
    "scripts/kaggle_runner.py",
)
DENYLIST_PARTS = {
    ".git",
    ".venv",
    "artifacts",
    "data/raw",
    "data/processed",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
DENYLIST_NAMES = {".env", "kaggle.json", "run_kaggle.py"}


class RunnerBuildError(RuntimeError):
    """Raised when a source snapshot could include unsafe or stale files."""


def archive_project(project_root: Path = PROJECT_ROOT) -> bytes:
    """Create a byte-for-byte deterministic gzip tar archive from the runtime allowlist."""
    entries = _collect_entries(project_root)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative, source in entries:
            payload = source.read_bytes()
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, filename="") as handle:
        handle.write(tar_buffer.getvalue())
    return compressed.getvalue()


def build_runner(project_root: Path = PROJECT_ROOT) -> str:
    archive = archive_project(project_root)
    return render_runner(
        base64.b64encode(archive).decode("ascii"),
        hashlib.sha256(archive).hexdigest(),
        len(archive),
    )


def write_runner(project_root: Path = PROJECT_ROOT, output: Path | None = None) -> Path:
    destination = output if output is not None else project_root / "docs" / "recorded-run" / "kaggle" / "run_kaggle.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_runner(project_root), encoding="utf-8")
    return destination


def check_runner(project_root: Path = PROJECT_ROOT, output: Path | None = None) -> bool:
    destination = output if output is not None else project_root / "docs" / "recorded-run" / "kaggle" / "run_kaggle.py"
    return destination.is_file() and destination.read_text(encoding="utf-8") == build_runner(project_root)


def archive_members(archive: bytes) -> tuple[str, ...]:
    """List embedded paths for tests without extracting caller-controlled data."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        return tuple(member.name for member in source.getmembers())


def _collect_entries(project_root: Path) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    for raw in ALLOWLIST:
        source = project_root / raw
        if source.is_file():
            candidates = [source]
        elif source.is_dir():
            candidates = sorted(
                (path for path in source.rglob("*") if path.is_file()), key=lambda path: path.as_posix()
            )
        else:
            raise RunnerBuildError(f"Kaggle runtime source is missing: {source}")
        for candidate in candidates:
            relative = candidate.relative_to(project_root)
            if _is_excluded_runtime_path(relative) or relative == Path("data/manifests/source.yaml"):
                continue
            _validate_runtime_path(relative)
            entries.append((relative, candidate))
    duplicate_names = [path for path, _ in entries]
    if len(set(duplicate_names)) != len(duplicate_names):
        raise RunnerBuildError("Kaggle runtime allowlist contains duplicate files")
    return sorted(entries, key=lambda item: item[0].as_posix())


def _is_excluded_runtime_path(relative: Path) -> bool:
    return relative.name in DENYLIST_NAMES or any(part in DENYLIST_PARTS for part in relative.parts)


def _validate_runtime_path(relative: Path) -> None:
    value = relative.as_posix()
    if _is_excluded_runtime_path(relative):
        raise RunnerBuildError(f"forbidden file in Kaggle source snapshot: {value}")
    lowered = value.lower()
    if "credential" in lowered or "token" in lowered or "secret" in lowered:
        raise RunnerBuildError(f"possible secret in Kaggle source snapshot: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the self-contained Kaggle runner")
    parser.add_argument("--check", action="store_true", help="fail when the committed runner is stale")
    args = parser.parse_args(argv)
    if args.check:
        if not check_runner():
            raise SystemExit("Kaggle runner is stale; run: uv run python scripts/build_kaggle_runner.py")
        print(RUNNER_PATH)
        return 0
    print(write_runner())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
