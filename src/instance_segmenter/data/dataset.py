"""Penn-Fudan Dataset implementation backed by fixed manifests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from instance_segmenter.data.manifest import (
    DatasetMetadata,
    ManifestError,
    ManifestRow,
    load_dataset_metadata,
    numeric_image_id,
    read_manifest,
)
from instance_segmenter.data.masks import rebuild_target_geometry, stack_instance_masks, validate_geometry
from instance_segmenter.data.schema import InstanceTarget
from instance_segmenter.data.transforms import Compose, InstanceTransform, RandomHorizontalFlip, Resize


class DatasetError(RuntimeError):
    """Raised when a source image no longer matches its manifest."""


class PennFudanDataset(Dataset[tuple[torch.Tensor, InstanceTarget]]):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        data_dir: str | Path,
        metadata: DatasetMetadata,
        *,
        transforms: InstanceTransform | None = None,
    ) -> None:
        if not rows:
            raise DatasetError("dataset split must not be empty")
        self.rows = tuple(rows)
        self.data_root = Path(data_dir) / metadata.dataset_root
        self.metadata = metadata
        self.transforms = transforms

    @classmethod
    def from_manifests(
        cls,
        manifest_dir: str | Path,
        split: str,
        *,
        data_dir: str | Path,
        training: bool = False,
        horizontal_flip: float = 0.0,
        image_size: tuple[int, int] | None = None,
        limit: int | None = None,
    ) -> PennFudanDataset:
        if split not in {"train", "valid", "test"}:
            raise DatasetError(f"unknown split {split!r}")
        metadata = load_dataset_metadata(manifest_dir)
        rows = read_manifest(Path(manifest_dir) / f"{split}.csv")
        if limit is not None:
            if limit <= 0:
                raise DatasetError("sample limit must be positive")
            rows = rows[:limit]
        transforms: list[InstanceTransform] = []
        if image_size is not None:
            transforms.append(Resize(image_size))
        if training and horizontal_flip:
            transforms.append(RandomHorizontalFlip(horizontal_flip))
        return cls(rows, data_dir, metadata, transforms=Compose(transforms))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, InstanceTarget]:
        row = self.rows[index]
        image_path = self.data_root / row.image_path
        mask_path = self.data_root / row.mask_path
        try:
            with Image.open(image_path) as source:
                image = pil_to_tensor(source.convert("RGB")).to(dtype=torch.float32).div(255.0)
            with Image.open(mask_path) as source:
                raw_mask = torch.as_tensor(np.asarray(source).copy())
        except OSError as exc:
            raise DatasetError(f"cannot read sample {row.image_id}: {exc}") from exc
        height, width = image.shape[-2:]
        if (width, height) != (row.width, row.height):
            raise DatasetError(f"image dimensions changed for {row.image_id}")
        if tuple(raw_mask.shape) != (height, width):
            raise DatasetError(f"mask dimensions changed for {row.image_id}")
        instance_ids = [value for value in torch.unique(raw_mask).tolist() if value != 0]
        masks = stack_instance_masks([(raw_mask == value).bool() for value in instance_ids], height, width)
        target: InstanceTarget = {
            "boxes": torch.empty((masks.shape[0], 4), dtype=torch.float32),
            "labels": torch.ones((masks.shape[0],), dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor([numeric_image_id(row.image_id)], dtype=torch.int64),
            "area": torch.empty((masks.shape[0],), dtype=torch.float32),
            "iscrowd": torch.zeros((masks.shape[0],), dtype=torch.int64),
        }
        target = rebuild_target_geometry(target)
        if target["masks"].shape[0] != row.instance_count:
            raise DatasetError(f"instance count changed for {row.image_id}")
        validate_geometry(target, height, width, self.metadata.label_schema)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def source_image_id(self, index: int) -> str:
        return self.rows[index].image_id


def build_pennfudan_dataset(
    manifest_dir: str | Path,
    split: str,
    *,
    data_dir: str | Path,
    training: bool,
    horizontal_flip: float,
    image_size: tuple[int, int] | None,
    limit: int | None,
) -> PennFudanDataset:
    """Provider-shaped factory used by the data registry."""
    try:
        return PennFudanDataset.from_manifests(
            manifest_dir,
            split,
            data_dir=data_dir,
            training=training,
            horizontal_flip=horizontal_flip,
            image_size=image_size,
            limit=limit,
        )
    except ManifestError as exc:
        raise DatasetError(str(exc)) from exc
