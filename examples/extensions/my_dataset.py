"""A custom dataset factory starts with the same target contract."""

from __future__ import annotations

from pathlib import Path

from instance_segmenter.data.dataset import PennFudanDataset


def build_dataset(
    *,
    manifest_dir: Path,
    split: str,
    data_dir: Path,
    training: bool,
    horizontal_flip: float,
    image_size: tuple[int, int] | None,
    limit: int | None,
) -> PennFudanDataset:
    """Replace this body to parse your image + instance-ID mask layout."""
    return PennFudanDataset.from_manifests(
        manifest_dir,
        split,
        data_dir=data_dir,
        training=training,
        horizontal_flip=horizontal_flip,
        image_size=image_size,
        limit=limit,
    )
