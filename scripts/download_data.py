"""Download and verify the official Penn-Fudan Pedestrian archive."""

from __future__ import annotations

import argparse
import datetime as datetime
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml

from instance_segmenter.data.manifest import PENN_FUDAN_DATASET_ROOT, sha256_file

PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
PENN_FUDAN_ARCHIVE_NAME = "PennFudanPed.zip"
PENN_FUDAN_ARCHIVE_SHA256 = "9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2"


class DownloadError(RuntimeError):
    """Raised when the official archive cannot be downloaded safely."""


def download_penn_fudan(data_dir: str | Path, manifest_dir: str | Path) -> Path:
    """Fetch once, checksum, safely extract, and record immutable source metadata."""
    root = Path(data_dir)
    archive = root / "downloads" / PENN_FUDAN_ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        _verify_archive(archive)
    else:
        _download_archive(archive)
    extracted = root / PENN_FUDAN_DATASET_ROOT
    if extracted.exists():
        _validate_extracted(extracted)
    else:
        _safe_extract(archive, root)
        _validate_extracted(extracted)
    _write_source_metadata(archive, manifest_dir)
    return extracted


def _download_archive(destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            request = urllib.request.Request(
                PENN_FUDAN_URL, headers={"User-Agent": "pytorch-instance-segmentation-lab"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, handle)
        _verify_archive(temporary)
        os.replace(temporary, destination)
    except (OSError, urllib.error.URLError) as exc:
        raise DownloadError(f"cannot download Penn-Fudan from {PENN_FUDAN_URL}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_archive(path: Path) -> None:
    actual = sha256_file(path)
    if actual != PENN_FUDAN_ARCHIVE_SHA256:
        raise DownloadError(
            f"Penn-Fudan archive checksum mismatch for {path}: expected {PENN_FUDAN_ARCHIVE_SHA256}, got {actual}"
        )


def _safe_extract(archive: Path, destination: Path) -> None:
    temporary_root = Path(tempfile.mkdtemp(dir=destination, prefix=".PennFudanPed."))
    try:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise DownloadError(f"unsafe archive member: {member.filename}")
            source.extractall(temporary_root)
        extracted = temporary_root / PENN_FUDAN_DATASET_ROOT
        _validate_extracted(extracted)
        target = destination / PENN_FUDAN_DATASET_ROOT
        if target.exists():
            raise DownloadError(f"will not overwrite existing extracted dataset {target}")
        os.replace(extracted, target)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DownloadError(f"cannot extract {archive}: {exc}") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _validate_extracted(root: Path) -> None:
    if not (root / "PNGImages").is_dir() or not (root / "PedMasks").is_dir():
        raise DownloadError(f"extracted Penn-Fudan data is incomplete at {root}")


def _write_source_metadata(archive: Path, manifest_dir: str | Path) -> None:
    output = Path(manifest_dir) / "source.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "dataset_name": "pennfudan",
        "source_url": PENN_FUDAN_URL,
        "archive_name": PENN_FUDAN_ARCHIVE_NAME,
        "archive_sha256": PENN_FUDAN_ARCHIVE_SHA256,
        "archive_bytes": archive.stat().st_size,
        "downloaded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the official Penn-Fudan Pedestrian dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args(argv)
    root = download_penn_fudan(args.data_dir, args.manifest_dir)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
