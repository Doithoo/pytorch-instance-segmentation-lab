"""Dataset provider registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from instance_segmenter.data.coco import build_coco_dataset
from instance_segmenter.data.dataset import build_pennfudan_dataset
from instance_segmenter.data.schema import InstanceTarget

DatasetFactory = Callable[..., Dataset[tuple[object, InstanceTarget]]]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    factory: DatasetFactory
    description: str


_REGISTRY: dict[str, DatasetSpec] = {
    "coco": DatasetSpec("coco", build_coco_dataset, "COCO instance JSON with polygon or RLE masks"),
    "pennfudan": DatasetSpec("pennfudan", build_pennfudan_dataset, "Penn-Fudan Pedestrian instance masks"),
}


def list_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset provider {name!r}; available: {', '.join(list_datasets())}") from exc


def build_dataset(
    name: str,
    manifest_dir: Path,
    split: str,
    *,
    data_dir: Path,
    training: bool,
    horizontal_flip: float,
    image_size: tuple[int, int] | None,
    limit: int | None,
) -> Dataset[tuple[object, InstanceTarget]]:
    return get_dataset_spec(name).factory(
        manifest_dir,
        split,
        data_dir=data_dir,
        training=training,
        horizontal_flip=horizontal_flip,
        image_size=image_size,
        limit=limit,
    )
