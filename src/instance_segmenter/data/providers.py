"""Provider protocol and configuration-aware dataset construction."""

from __future__ import annotations

from torch.utils.data import Dataset

from instance_segmenter.config import DataConfig
from instance_segmenter.data.extensions import load_external_dataset
from instance_segmenter.data.registry import DatasetSpec, build_dataset, get_dataset_spec, list_datasets
from instance_segmenter.data.schema import InstanceTarget


def build_configured_dataset(
    config: DataConfig,
    split: str,
    *,
    training: bool,
    limit: int | None,
) -> Dataset[tuple[object, InstanceTarget]]:
    if config.factory is not None:
        return load_external_dataset(
            config.factory,
            config.manifest_dir,
            split,
            data_dir=config.root,
            training=training,
            horizontal_flip=config.horizontal_flip if training else 0.0,
            image_size=config.image_size,
            limit=limit,
        )
    return build_dataset(
        config.provider,
        config.manifest_dir,
        split,
        data_dir=config.root,
        training=training,
        horizontal_flip=config.horizontal_flip if training else 0.0,
        image_size=config.image_size,
        limit=limit,
    )


__all__ = ["DatasetSpec", "build_configured_dataset", "build_dataset", "get_dataset_spec", "list_datasets"]
