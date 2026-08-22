"""Fixed Penn-Fudan manifests, dataset identity, and verification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from instance_segmenter.data.masks import decode_instance_mask
from instance_segmenter.data.schema import DEFAULT_LABEL_SCHEMA, LabelSchema

PENN_FUDAN_DATASET_ROOT = "PennFudanPed"
PENN_FUDAN_SPLIT_COUNTS = {"train": 136, "valid": 17, "test": 17}
MANIFEST_FIELDS = (
    "image_id",
    "image_path",
    "mask_path",
    "width",
    "height",
    "instance_count",
    "image_sha256",
    "mask_sha256",
)


class ManifestError(RuntimeError):
    """Raised when source data and a fixed manifest disagree."""


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    image_path: str
    mask_path: str
    width: int
    height: int
    instance_count: int
    image_sha256: str
    mask_sha256: str

    @classmethod
    def from_mapping(cls, raw: dict[str, str]) -> ManifestRow:
        if tuple(raw) != MANIFEST_FIELDS:
            raise ManifestError(f"manifest columns must be {MANIFEST_FIELDS}")
        try:
            row = cls(
                image_id=raw["image_id"],
                image_path=raw["image_path"],
                mask_path=raw["mask_path"],
                width=int(raw["width"]),
                height=int(raw["height"]),
                instance_count=int(raw["instance_count"]),
                image_sha256=raw["image_sha256"],
                mask_sha256=raw["mask_sha256"],
            )
        except (KeyError, ValueError) as exc:
            raise ManifestError(f"invalid manifest row: {exc}") from exc
        if not row.image_id or row.width <= 0 or row.height <= 0 or row.instance_count < 0:
            raise ManifestError(f"invalid manifest row for {row.image_id!r}")
        return row

    def to_mapping(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_name: str
    dataset_root: str
    label_schema: LabelSchema
    split_counts: dict[str, int]
    split_hashes: dict[str, str]
    identity: str
    image_width_range: tuple[int, int]
    image_height_range: tuple[int, int]
    instance_count_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_root": self.dataset_root,
            "label_schema": self.label_schema.to_dict(),
            "split_counts": self.split_counts,
            "split_hashes": self.split_hashes,
            "identity": self.identity,
            "image_width_range": list(self.image_width_range),
            "image_height_range": list(self.image_height_range),
            "instance_count_range": list(self.instance_count_range),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetMetadata:
        required = {
            "dataset_name",
            "dataset_root",
            "label_schema",
            "split_counts",
            "split_hashes",
            "identity",
            "image_width_range",
            "image_height_range",
            "instance_count_range",
        }
        if set(raw) != required:
            raise ManifestError(f"dataset metadata fields must be {sorted(required)}")
        try:
            metadata = cls(
                dataset_name=str(raw["dataset_name"]),
                dataset_root=str(raw["dataset_root"]),
                label_schema=LabelSchema.from_dict(raw["label_schema"]),
                split_counts={str(key): int(value) for key, value in raw["split_counts"].items()},
                split_hashes={str(key): str(value) for key, value in raw["split_hashes"].items()},
                identity=str(raw["identity"]),
                image_width_range=tuple(raw["image_width_range"]),
                image_height_range=tuple(raw["image_height_range"]),
                instance_count_range=tuple(raw["instance_count_range"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"invalid dataset metadata: {exc}") from exc
        if metadata.split_counts.keys() != PENN_FUDAN_SPLIT_COUNTS.keys():
            raise ManifestError("metadata must contain train, valid, and test counts")
        return metadata


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_image_id(image_id: str) -> int:
    """Produce a reproducible signed 63-bit ID accepted by torchvision metrics."""
    return int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def prepare_penn_fudan(
    data_dir: str | Path,
    manifest_dir: str | Path,
    *,
    expected_total: int | None = 170,
) -> DatasetMetadata:
    """Validate Penn-Fudan sources and atomically write deterministic manifests."""
    data_root = Path(data_dir) / PENN_FUDAN_DATASET_ROOT
    image_dir = data_root / "PNGImages"
    mask_dir = data_root / "PedMasks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ManifestError(f"expected {image_dir} and {mask_dir}; run download_data.py first")
    pairs = _discover_pairs(data_root, image_dir, mask_dir)
    if expected_total is not None and len(pairs) != expected_total:
        raise ManifestError(f"expected {expected_total} Penn-Fudan pairs, found {len(pairs)}")
    splits = _split_pairs(pairs, expected_total=expected_total)
    output = Path(manifest_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_csv(output / f"{split}.csv", rows)
    split_hashes = {split: sha256_file(output / f"{split}.csv") for split in PENN_FUDAN_SPLIT_COUNTS}
    all_rows = tuple(row for rows in splits.values() for row in rows)
    metadata = DatasetMetadata(
        dataset_name="pennfudan",
        dataset_root=PENN_FUDAN_DATASET_ROOT,
        label_schema=DEFAULT_LABEL_SCHEMA,
        split_counts={split: len(rows) for split, rows in splits.items()},
        split_hashes=split_hashes,
        identity=_identity(splits),
        image_width_range=(min(row.width for row in all_rows), max(row.width for row in all_rows)),
        image_height_range=(min(row.height for row in all_rows), max(row.height for row in all_rows)),
        instance_count_range=(min(row.instance_count for row in all_rows), max(row.instance_count for row in all_rows)),
    )
    _write_yaml(output / "dataset.yaml", metadata.to_dict())
    return metadata


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ManifestError(f"manifest {manifest_path} has no header")
            rows = [ManifestRow.from_mapping(dict(row)) for row in reader]
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not rows:
        raise ManifestError(f"manifest {manifest_path} is empty")
    if len({row.image_id for row in rows}) != len(rows):
        raise ManifestError(f"manifest {manifest_path} has duplicate image IDs")
    return rows


def load_dataset_metadata(manifest_dir: str | Path) -> DatasetMetadata:
    path = Path(manifest_dir) / "dataset.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read dataset metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("dataset metadata root must be a mapping")
    return DatasetMetadata.from_dict(raw)


def verify_prepared_data(data_dir: str | Path, manifest_dir: str | Path) -> DatasetMetadata:
    """Revalidate source bytes, image dimensions, and masks against all manifests."""
    metadata = load_dataset_metadata(manifest_dir)
    root = Path(data_dir) / metadata.dataset_root
    total_ids: set[str] = set()
    for split, expected_count in metadata.split_counts.items():
        path = Path(manifest_dir) / f"{split}.csv"
        if sha256_file(path) != metadata.split_hashes[split]:
            raise ManifestError(f"manifest hash mismatch for {split}: {path}")
        rows = read_manifest(path)
        if len(rows) != expected_count:
            raise ManifestError(f"manifest count mismatch for {split}")
        for row in rows:
            if row.image_id in total_ids:
                raise ManifestError(f"duplicate image ID across splits: {row.image_id}")
            total_ids.add(row.image_id)
            _verify_row(root, row)
    if metadata.dataset_name == "pennfudan" and metadata.split_counts != PENN_FUDAN_SPLIT_COUNTS:
        raise ManifestError(f"Penn-Fudan split counts must be {PENN_FUDAN_SPLIT_COUNTS}")
    return metadata


def _discover_pairs(data_root: Path, image_dir: Path, mask_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for image_path in sorted(image_dir.glob("*.png"), key=lambda path: path.name):
        image_id = image_path.stem
        mask_path = mask_dir / f"{image_id}_mask.png"
        if not mask_path.is_file():
            raise ManifestError(f"missing mask for {image_path.name}: {mask_path}")
        try:
            with Image.open(image_path) as image:
                width, height = image.convert("RGB").size
            with Image.open(mask_path) as mask:
                if mask.size != (width, height):
                    raise ManifestError(f"mask size does not match image for {image_id}")
        except OSError as exc:
            raise ManifestError(f"cannot decode {image_id}: {exc}") from exc
        masks = decode_instance_mask(mask_path)
        if not masks:
            raise ManifestError(f"{image_id} has no pedestrian instances")
        rows.append(
            ManifestRow(
                image_id=image_id,
                image_path=image_path.relative_to(data_root).as_posix(),
                mask_path=mask_path.relative_to(data_root).as_posix(),
                width=width,
                height=height,
                instance_count=len(masks),
                image_sha256=sha256_file(image_path),
                mask_sha256=sha256_file(mask_path),
            )
        )
    if not rows:
        raise ManifestError(f"no PNG images found in {image_dir}")
    orphan_masks = {path.stem.removesuffix("_mask") for path in mask_dir.glob("*_mask.png")} - {
        row.image_id for row in rows
    }
    if orphan_masks:
        raise ManifestError(f"masks without matching images: {sorted(orphan_masks)[:3]}")
    return rows


def _split_pairs(rows: list[ManifestRow], *, expected_total: int | None) -> dict[str, list[ManifestRow]]:
    if expected_total == 170:
        boundaries = (136, 153)
    else:
        total = len(rows)
        train = max(1, int(total * 0.8))
        valid = max(1, int(total * 0.1))
        if train + valid >= total:
            raise ManifestError("need at least three source pairs to create non-empty splits")
        boundaries = (train, train + valid)
    train_end, valid_end = boundaries
    splits = {"train": rows[:train_end], "valid": rows[train_end:valid_end], "test": rows[valid_end:]}
    if any(not values for values in splits.values()):
        raise ManifestError("all dataset splits must be non-empty")
    return splits


def _verify_row(data_root: Path, row: ManifestRow) -> None:
    image_path = data_root / row.image_path
    mask_path = data_root / row.mask_path
    if not image_path.is_file() or not mask_path.is_file():
        raise ManifestError(f"source file missing for {row.image_id}")
    if sha256_file(image_path) != row.image_sha256 or sha256_file(mask_path) != row.mask_sha256:
        raise ManifestError(f"source file hash mismatch for {row.image_id}")
    try:
        with Image.open(image_path) as image:
            if image.size != (row.width, row.height):
                raise ManifestError(f"image dimensions changed for {row.image_id}")
    except OSError as exc:
        raise ManifestError(f"cannot decode image {row.image_id}: {exc}") from exc
    masks = decode_instance_mask(mask_path)
    if len(masks) != row.instance_count:
        raise ManifestError(f"instance count changed for {row.image_id}")


def _identity(splits: dict[str, list[ManifestRow]]) -> str:
    raw = {split: [row.to_mapping() for row in rows] for split, rows in splits.items()}
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_file(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(row.to_mapping() for row in rows)
        os.replace(temporary, path)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_file(path) as temporary:
        temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)


class _temporary_file:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.path: Path | None = None

    def __enter__(self) -> Path:
        descriptor, raw_path = tempfile.mkstemp(dir=self.output.parent, prefix=f".{self.output.name}.", suffix=".tmp")
        os.close(descriptor)
        self.path = Path(raw_path)
        return self.path

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)
