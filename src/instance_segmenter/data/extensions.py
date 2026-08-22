"""Load an explicit external Dataset factory without modifying the registry."""

from __future__ import annotations

import importlib
from pathlib import Path

from torch.utils.data import Dataset

from instance_segmenter.data.schema import InstanceTarget


def load_external_dataset(
    factory_path: str,
    manifest_dir: Path,
    split: str,
    *,
    data_dir: Path,
    training: bool,
    horizontal_flip: float,
    image_size: tuple[int, int] | None,
    limit: int | None,
) -> Dataset[tuple[object, InstanceTarget]]:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("external dataset factory must use module.path:callable_name")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"external dataset factory {factory_path!r} is not callable")
    dataset = factory(
        manifest_dir=manifest_dir,
        split=split,
        data_dir=data_dir,
        training=training,
        horizontal_flip=horizontal_flip,
        image_size=image_size,
        limit=limit,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("external dataset factory must return torch.utils.data.Dataset")
    return dataset
