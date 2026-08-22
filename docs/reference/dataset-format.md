# Dataset Formats

## Penn-Fudan instance IDs

The built-in Penn-Fudan provider expects `PennFudanPed/PNGImages/<id>.png` and `PennFudanPed/PedMasks/<id>_mask.png`. Mask value 0 is background; each positive integer identifies one independent person. Preparation verifies dimensions and source SHA-256 hashes, then writes source-stratified fixed manifests.

## COCO instance JSON

COCO preparation supports polygon lists and compressed or uncompressed RLE. Categories are sorted by original category ID and mapped to contiguous model labels starting at 1; the mapping is stored in `dataset.yaml`. Images with no annotations are valid.

```bash
uv run instance-segment prepare-coco \
  --data-dir data/coco \
  --manifest-dir data/coco-manifests \
  --train-annotations annotations/instances_train.json \
  --valid-annotations annotations/instances_valid.json \
  --test-annotations annotations/instances_test.json
```

Annotation and image paths must stay inside `data-dir`. Category definitions must agree across all three files. Preparation records annotation/image hashes and dimensions; `verify-data` checks them before training.

## Runtime target

Both providers return float32 `image[C,H,W]` in `[0,1]` and a target containing float32 half-open `boxes[N,4]`, int64 `labels[N]`, bool `masks[N,H,W]`, int64 `image_id[1]`, float32 mask-pixel `area[N]`, and int64 `iscrowd[N]`. All instance fields have the same `N`; `N=0` is supported for COCO.
