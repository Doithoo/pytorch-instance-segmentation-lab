from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from instance_segmenter.data.coco import build_coco_dataset, prepare_coco_instances
from instance_segmenter.data.manifest import verify_prepared_data


def _write_split(root: Path, split: str) -> Path:
    image_dir = root / "images" / split
    image_dir.mkdir(parents=True)
    first = np.zeros((6, 8, 3), dtype=np.uint8)
    second = np.full((6, 8, 3), 100, dtype=np.uint8)
    Image.fromarray(first).save(image_dir / "one.png")
    Image.fromarray(second).save(image_dir / "empty.png")
    binary = np.zeros((6, 8), dtype=np.uint8)
    binary[1:4, 4:7] = 1
    rle = mask_utils.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("ascii")
    payload = {
        "images": [
            {"id": 1, "file_name": f"images/{split}/one.png", "width": 8, "height": 6},
            {"id": 2, "file_name": f"images/{split}/empty.png", "width": 8, "height": 6},
        ],
        "categories": [{"id": 3, "name": "cat"}, {"id": 7, "name": "dog"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "iscrowd": 0,
                "segmentation": [[1, 1, 3, 1, 3, 4, 1, 4]],
            },
            {"id": 2, "image_id": 1, "category_id": 7, "iscrowd": 1, "segmentation": rle},
        ],
    }
    annotation = root / "annotations" / f"instances_{split}.json"
    annotation.parent.mkdir(exist_ok=True)
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    return annotation


def test_coco_polygon_rle_multiclass_and_empty_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "coco"
    manifest_dir = tmp_path / "manifests"
    annotations = {split: _write_split(data_dir, split) for split in ("train", "valid", "test")}
    metadata = prepare_coco_instances(data_dir, manifest_dir, annotations)
    assert metadata.label_schema.num_classes == 3
    assert metadata.instance_count_range == (0, 2)
    assert verify_prepared_data(data_dir, manifest_dir).identity == metadata.identity

    dataset = build_coco_dataset(
        manifest_dir,
        "train",
        data_dir=data_dir,
        training=True,
        horizontal_flip=1.0,
        image_size=(12, 16),
        limit=None,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    populated = next(target for _, target in samples if target["labels"].numel())
    empty = next(target for _, target in samples if not target["labels"].numel())
    assert set(populated["labels"].tolist()) == {1, 2}
    assert populated["masks"].dtype == torch.bool
    assert populated["masks"].shape == (2, 12, 16)
    assert empty["masks"].shape == (0, 12, 16)
