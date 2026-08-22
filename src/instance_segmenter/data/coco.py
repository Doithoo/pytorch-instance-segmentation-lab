"""COCO instance JSON preparation and dataset provider."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from instance_segmenter.data.manifest import (
    MANIFEST_FORMAT_VERSION,
    DatasetMetadata,
    ManifestError,
    ManifestRow,
    load_dataset_metadata,
    numeric_image_id,
    read_manifest,
    sha256_file,
)
from instance_segmenter.data.masks import rebuild_target_geometry, stack_instance_masks, validate_geometry
from instance_segmenter.data.schema import ClassDefinition, InstanceTarget, LabelSchema
from instance_segmenter.data.transforms import Compose, InstanceTransform, RandomHorizontalFlip, Resize

_COCO_COLORS = (
    (36, 180, 99),
    (45, 125, 210),
    (230, 120, 50),
    (190, 70, 150),
    (225, 80, 80),
    (120, 150, 50),
)


class CocoDatasetError(RuntimeError):
    """Raised when COCO JSON or an image violates the prepared contract."""


def prepare_coco_instances(
    data_dir: str | Path,
    manifest_dir: str | Path,
    annotation_files: Mapping[str, str | Path],
) -> DatasetMetadata:
    """Prepare fixed manifests for train/valid/test COCO instance JSON files."""
    if set(annotation_files) != {"train", "valid", "test"}:
        raise ManifestError("COCO preparation requires train, valid, and test annotation files")
    root = Path(data_dir).resolve()
    output = Path(manifest_dir)
    output.mkdir(parents=True, exist_ok=True)
    category_signature: tuple[tuple[int, str], ...] | None = None
    category_map: dict[int, int] = {}
    relative_annotations: dict[str, str] = {}
    split_rows: dict[str, list[ManifestRow]] = {}

    for split in ("train", "valid", "test"):
        candidate = Path(annotation_files[split])
        annotation_path = _inside_root(root, candidate if candidate.is_absolute() else root / candidate)
        raw = _read_coco_json(annotation_path)
        categories = _categories(raw)
        _validate_coco_relations(raw, {item["id"] for item in categories})
        signature = tuple((item["id"], item["name"]) for item in categories)
        if category_signature is None:
            category_signature = signature
            category_map = {category_id: index for index, (category_id, _) in enumerate(signature, start=1)}
        elif signature != category_signature:
            raise ManifestError("COCO category definitions differ across splits")
        relative_annotation = annotation_path.relative_to(root).as_posix()
        relative_annotations[split] = relative_annotation
        rows = _manifest_rows(root, raw, relative_annotation, annotation_path, split)
        if not rows:
            raise ManifestError(f"COCO {split} split is empty")
        split_rows[split] = rows
        _write_manifest(output / f"{split}.csv", rows)

    assert category_signature is not None
    class_definitions = [ClassDefinition(0, "background", (32, 32, 32))]
    used_colors = {(32, 32, 32)}
    for index, (source_id, name) in enumerate(category_signature, start=1):
        color = _category_color(source_id, name, used_colors)
        used_colors.add(color)
        class_definitions.append(ClassDefinition(index, name, color))
    classes = tuple(class_definitions)
    all_rows = [row for rows in split_rows.values() for row in rows]
    split_hashes = {split: sha256_file(output / f"{split}.csv") for split in split_rows}
    identity_payload = {
        "format": MANIFEST_FORMAT_VERSION,
        "annotations": {split: sha256_file(root / path) for split, path in relative_annotations.items()},
        "rows": {split: [row.to_mapping() for row in rows] for split, rows in split_rows.items()},
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = DatasetMetadata(
        dataset_name="coco",
        dataset_root=".",
        label_schema=LabelSchema(classes),
        split_counts={split: len(rows) for split, rows in split_rows.items()},
        split_hashes=split_hashes,
        identity=identity,
        image_width_range=(min(row.width for row in all_rows), max(row.width for row in all_rows)),
        image_height_range=(min(row.height for row in all_rows), max(row.height for row in all_rows)),
        instance_count_range=(
            min(row.instance_count for row in all_rows),
            max(row.instance_count for row in all_rows),
        ),
        manifest_format_version=MANIFEST_FORMAT_VERSION,
        split_strategy="official-coco-json-v1",
        split_seed=0,
        provider_metadata={
            "annotation_files": relative_annotations,
            "category_id_map": {str(source): target for source, target in category_map.items()},
        },
    )
    _write_yaml(output / "dataset.yaml", metadata.to_dict())
    return metadata


class CocoInstanceDataset(Dataset[tuple[torch.Tensor, InstanceTarget]]):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        data_dir: str | Path,
        metadata: DatasetMetadata,
        split: str,
        *,
        transforms: InstanceTransform | None = None,
    ) -> None:
        if not rows:
            raise CocoDatasetError("dataset split must not be empty")
        annotation_files = metadata.provider_metadata.get("annotation_files")
        category_id_map = metadata.provider_metadata.get("category_id_map")
        if not isinstance(annotation_files, dict) or not isinstance(category_id_map, dict):
            raise CocoDatasetError("COCO metadata misses annotation_files or category_id_map")
        annotation_relative = annotation_files.get(split)
        if not isinstance(annotation_relative, str):
            raise CocoDatasetError(f"COCO metadata misses {split} annotations")
        self.rows = tuple(rows)
        self.data_root = Path(data_dir) / metadata.dataset_root
        self.metadata = metadata
        self.transforms = transforms
        raw = _read_coco_json(self.data_root / annotation_relative)
        annotations: dict[str, list[dict[str, Any]]] = {}
        for annotation in raw.get("annotations", []):
            if isinstance(annotation, dict):
                annotations.setdefault(str(annotation.get("image_id")), []).append(annotation)
        self.annotations = annotations
        self.category_id_map = {int(source): int(target) for source, target in category_id_map.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, InstanceTarget]:
        row = self.rows[index]
        try:
            with Image.open(self.data_root / row.image_path) as source:
                image = pil_to_tensor(source.convert("RGB")).to(torch.float32).div(255.0)
        except OSError as exc:
            raise CocoDatasetError(f"cannot read sample {row.image_id}: {exc}") from exc
        height, width = image.shape[-2:]
        if (width, height) != (row.width, row.height):
            raise CocoDatasetError(f"image dimensions changed for {row.image_id}")
        source_id = row.image_id.split(":", 1)[-1]
        masks: list[torch.Tensor] = []
        labels: list[int] = []
        crowd: list[int] = []
        for annotation in self.annotations.get(source_id, []):
            category_id = int(annotation.get("category_id", -1))
            if category_id not in self.category_id_map:
                raise CocoDatasetError(f"unknown category {category_id} in image {source_id}")
            masks.append(_decode_segmentation(annotation.get("segmentation"), height, width))
            labels.append(self.category_id_map[category_id])
            crowd.append(int(annotation.get("iscrowd", 0)))
        mask_tensor = stack_instance_masks(masks, height, width)
        target: InstanceTarget = {
            "boxes": torch.empty((len(masks), 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": mask_tensor,
            "image_id": torch.tensor([numeric_image_id(row.image_id)], dtype=torch.int64),
            "area": torch.empty((len(masks),), dtype=torch.float32),
            "iscrowd": torch.tensor(crowd, dtype=torch.int64),
        }
        target = rebuild_target_geometry(target)
        if target["masks"].shape[0] != row.instance_count:
            raise CocoDatasetError(f"instance count changed or contains empty masks for {row.image_id}")
        validate_geometry(target, height, width, self.metadata.label_schema)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_coco_dataset(
    manifest_dir: str | Path,
    split: str,
    *,
    data_dir: str | Path,
    training: bool,
    horizontal_flip: float,
    image_size: tuple[int, int] | None,
    limit: int | None,
) -> CocoInstanceDataset:
    if split not in {"train", "valid", "test"}:
        raise CocoDatasetError(f"unknown split {split!r}")
    metadata = load_dataset_metadata(manifest_dir)
    if metadata.dataset_name != "coco":
        raise CocoDatasetError("prepared metadata is not a COCO dataset")
    rows = read_manifest(Path(manifest_dir) / f"{split}.csv")
    if limit is not None:
        rows = rows[:limit]
    transforms: list[InstanceTransform] = []
    if image_size is not None:
        transforms.append(Resize(image_size))
    if training and horizontal_flip:
        transforms.append(RandomHorizontalFlip(horizontal_flip))
    return CocoInstanceDataset(rows, data_dir, metadata, split, transforms=Compose(transforms))


def _read_coco_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read COCO annotations {path}: {exc}") from exc
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("images"), list)
        or not isinstance(raw.get("categories"), list)
        or not isinstance(raw.get("annotations", []), list)
    ):
        raise ManifestError(f"COCO annotations require images, categories, and annotations arrays: {path}")
    return raw


def _categories(raw: dict[str, Any]) -> list[dict[str, Any]]:
    categories = raw["categories"]
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and not isinstance(item.get("id"), bool)
        and item.get("name")
        for item in categories
    ):
        raise ManifestError("COCO categories require integer id and non-empty name")
    ordered = sorted(categories, key=lambda item: item["id"])
    if len({item["id"] for item in ordered}) != len(ordered):
        raise ManifestError("COCO category ids must be unique")
    return ordered


def _validate_coco_relations(raw: dict[str, Any], category_ids: set[int]) -> None:
    image_ids = [item.get("id") for item in raw["images"] if isinstance(item, dict)]
    if (
        len(image_ids) != len(raw["images"])
        or any(isinstance(value, bool) or not isinstance(value, int | str) for value in image_ids)
        or len(set(image_ids)) != len(image_ids)
    ):
        raise ManifestError("COCO image ids must be present, scalar, and unique")
    image_id_set = set(image_ids)
    annotation_ids: list[object] = []
    for annotation in raw.get("annotations", []):
        if not isinstance(annotation, dict):
            raise ManifestError("COCO annotations must be objects")
        annotation_ids.append(annotation.get("id"))
        if annotation.get("image_id") not in image_id_set:
            raise ManifestError("COCO annotation references an unknown image")
        if annotation.get("category_id") not in category_ids:
            raise ManifestError("COCO annotation references an unknown category")
        if "segmentation" not in annotation:
            raise ManifestError("COCO instance annotation misses segmentation")
    if any(isinstance(value, bool) or not isinstance(value, int | str) for value in annotation_ids) or len(
        set(annotation_ids)
    ) != len(annotation_ids):
        raise ManifestError("COCO annotation ids must be present, scalar, and unique")


def _manifest_rows(
    root: Path,
    raw: dict[str, Any],
    annotation_relative: str,
    annotation_path: Path,
    split: str,
) -> list[ManifestRow]:
    counts: dict[str, int] = {}
    for annotation in raw.get("annotations", []):
        if isinstance(annotation, dict):
            key = str(annotation.get("image_id"))
            counts[key] = counts.get(key, 0) + 1
    annotation_hash = sha256_file(annotation_path)
    rows: list[ManifestRow] = []
    for item in sorted(raw["images"], key=lambda image: str(image.get("id"))):
        if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("file_name"), str):
            raise ManifestError("COCO images require id and file_name")
        image_path = _inside_root(root, root / item["file_name"])
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            raise ManifestError(f"cannot read COCO image {image_path}: {exc}") from exc
        if int(item.get("width", width)) != width or int(item.get("height", height)) != height:
            raise ManifestError(f"COCO image dimensions disagree with JSON: {image_path}")
        source_id = str(item["id"])
        rows.append(
            ManifestRow(
                image_id=f"{split}:{source_id}",
                image_path=image_path.relative_to(root).as_posix(),
                mask_path=annotation_relative,
                width=width,
                height=height,
                instance_count=counts.get(source_id, 0),
                image_sha256=sha256_file(image_path),
                mask_sha256=annotation_hash,
            )
        )
    return rows


def _decode_segmentation(segmentation: object, height: int, width: int) -> torch.Tensor:
    try:
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            rle = (
                mask_utils.frPyObjects(segmentation, height, width)
                if isinstance(segmentation.get("counts"), list)
                else segmentation
            )
        else:
            raise CocoDatasetError("COCO annotation segmentation must be polygon or RLE")
        decoded = mask_utils.decode(rle)
    except (TypeError, ValueError) as exc:
        raise CocoDatasetError(f"cannot decode COCO segmentation: {exc}") from exc
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise CocoDatasetError("decoded COCO mask dimensions do not match image")
    return torch.from_numpy(np.asarray(decoded, dtype=np.bool_).copy())


def _inside_root(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"COCO path must stay inside data root {root}: {path}") from exc
    return resolved


def _category_color(source_id: int, name: str, used: set[tuple[int, int, int]]) -> tuple[int, int, int]:
    preferred = _COCO_COLORS[len(used) - 1] if len(used) - 1 < len(_COCO_COLORS) else None
    if preferred is not None and preferred not in used:
        return preferred
    nonce = 0
    while True:
        digest = hashlib.sha256(f"{source_id}:{name}:{nonce}".encode()).digest()
        color = (digest[0], digest[1], digest[2])
        if color not in used:
            return color
        nonce += 1


def _write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].to_mapping()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(row.to_mapping() for row in rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_yaml(path: Path, raw: dict[str, object]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
