# Dataset Formats

[中文](dataset-format.zh-CN.md) | [Documentation index](../README.md)

Preparation creates a manifest directory containing dataset metadata and one CSV per split. The dataset identity is derived from the prepared metadata and split hashes; training and evaluation use it to reject changed data.

## Penn-Fudan Layout

```text
<data-dir>/PennFudanPed/PNGImages/<id>.png
<data-dir>/PennFudanPed/PedMasks/<id>_mask.png
```

Mask value `0` is background and every positive integer is one instance ID. Protocol-v2 preparation creates 136/17/17 train/valid/test rows using source-stratified SHA-256 ordering. See [Penn-Fudan reference](penn-fudan.md) for composition and provenance.

## Manifest Rows

The prepared CSVs contain the paths and integrity data required by the providers. Treat them as generated files. A row identifies an image, its instance mask or annotation record, dimensions, instance count, and source hashes. `dataset.yaml` stores format version, provider, label schema, split hashes, counts, and dataset identity; `source.yaml` stores download provenance where applicable.

## COCO Instance JSON

`prepare-coco` requires `images`, `annotations`, and `categories` in each JSON. The three category definitions must agree. Supported segmentation values are polygon lists and compressed or uncompressed RLE. Paths resolved from image records and annotation inputs must remain within `--data-dir`.

Categories are ordered by original category ID and mapped to contiguous model IDs starting at `1`; `0` remains background. The mapping is written to `dataset.yaml`. Crowd values and images without annotations are retained. Annotation and image dimensions are checked during preparation and source hashes are recorded for verification.

```bash
uv run instance-segment prepare-coco \
  --data-dir data/coco --manifest-dir data/coco-manifests \
  --train-annotations annotations/train.json \
  --valid-annotations annotations/valid.json \
  --test-annotations annotations/test.json
```

## Runtime Target

Both built-in providers return float32 images in `[0,1]` with the exact target contract:

```text
boxes   float32 [N,4]
labels  int64   [N]
masks   bool    [N,H,W]
image_id int64  [1]
area    float32 [N]
iscrowd int64   [N]
```

All instance fields share `N`; `N=0` is supported for COCO. After any resize or horizontal flip, boxes and area are derived from the transformed masks. Validate a custom provider with `validate_instance_target` and the extension example before a long run.
