# Data and Instances

[中文](02-data-and-instances.zh-CN.md) | [Documentation index](../README.md)

The default example is Penn-Fudan Pedestrian. The download script verifies the official archive before extraction; preparation then creates deterministic manifests rather than relying on directory ordering.

## Prepare Penn-Fudan

```bash
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment verify-data --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment inspect-data --data-dir data/raw --manifest-dir data/manifests --split train
uv run python scripts/preview_dataset.py --output artifacts/dataset-preview.png
```

The committed protocol-v2 manifests contain 170 images split into 136 train, 17 valid, and 17 test rows. The split is source-stratified between Fudan and Penn and is recorded in `data/manifests/dataset.yaml`. Every row includes dimensions, instance count, and image/mask SHA-256 values. Do not hand-edit a CSV after preparation; its identity is part of checkpoint compatibility.

## Read An Instance-ID Mask

Penn-Fudan masks use pixel value `0` for background. Each positive integer identifies one object. The provider creates a boolean mask for each positive ID, computes its half-open `xyxy` box and pixel area, and assigns the foreground label `1` (`person`). Sparse IDs do not create empty objects and touching IDs remain separate.

Geometric transforms apply to the image and masks together. Boxes and areas are derived again after resizing or horizontal flipping, so stale boxes cannot silently survive a transform. The collate function returns lists because images and instance counts are variable.

## COCO Alternative

For polygon or RLE annotations, use three split-specific COCO instance JSON files under one data root:

```bash
uv run instance-segment prepare-coco \
  --data-dir data/coco --manifest-dir data/coco-manifests \
  --train-annotations annotations/instances_train.json \
  --valid-annotations annotations/instances_valid.json \
  --test-annotations annotations/instances_test.json
uv run instance-segment verify-data --data-dir data/coco --manifest-dir data/coco-manifests
```

Category IDs are sorted and remapped to contiguous model labels starting at `1`; the mapping is persisted in `dataset.yaml`. Polygon lists, compressed/uncompressed RLE, multiclass objects, crowd flags, and empty images are supported. Annotation and image paths must remain inside `--data-dir`.

See [Dataset formats](../reference/dataset-format.md) for CSV columns, COCO consistency rules, and target dtypes.
