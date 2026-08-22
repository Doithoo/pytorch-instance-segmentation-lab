"""Validate prepared instance data and report useful split statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from instance_segmenter.data.manifest import load_dataset_metadata, read_manifest, verify_prepared_data


def inspect_prepared_data(data_dir: str | Path, manifest_dir: str | Path, split: str = "train") -> dict[str, Any]:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"unknown split {split!r}")
    metadata = verify_prepared_data(data_dir, manifest_dir)
    rows = read_manifest(Path(manifest_dir) / f"{split}.csv")
    return {
        "dataset": metadata.dataset_name,
        "identity": metadata.identity,
        "split": split,
        "images": len(rows),
        "instances": sum(row.instance_count for row in rows),
        "image_width_range": [min(row.width for row in rows), max(row.width for row in rows)],
        "image_height_range": [min(row.height for row in rows), max(row.height for row in rows)],
        "instance_count_range": [min(row.instance_count for row in rows), max(row.instance_count for row in rows)],
        "label_schema": load_dataset_metadata(manifest_dir).label_schema.to_dict(),
    }
