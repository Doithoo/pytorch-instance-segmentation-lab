# Using Your Data

[中文](using-your-data.zh-CN.md) | [Documentation index](../README.md)

Use the built-in COCO route when your annotations can be expressed as instance polygons or RLE. It preserves the most useful standard metadata and avoids writing a provider before the dataset contract is understood.

## COCO Route

Put images and the three split-specific instance JSON files below one data root. Run `prepare-coco` with explicit annotation paths, then verify and inspect each split. The category definitions must agree across all JSON files; image and annotation paths must stay inside the data root.

COCO preparation supports polygon lists, compressed/uncompressed RLE, multiple foreground classes, `iscrowd`, and images with no annotations. Source category IDs are mapped to contiguous model IDs and the mapping is persisted in `dataset.yaml`. Set `model.num_classes` to the resulting schema count, including background.

## Penn-Fudan-Style Masks

For an indexed PNG layout, use `0` for background and a distinct positive integer for every object. Do not use one shared value for all foreground pixels: that would erase instance boundaries. Every positive region must have nonzero area and fit the image dimensions.

## Custom Provider

Use `data.factory=module:callable` only when COCO and Penn-Fudan do not fit. The callable receives:

```text
manifest_dir, split, data_dir, training,
horizontal_flip, image_size, limit
```

The custom dataset still needs a manifest directory with `dataset.yaml` and hashed split CSVs because checkpoint validation and run comparison depend on those identities. Start from the [`my_dataset.py` extension example on GitHub](https://github.com/Doithoo/pytorch-instance-segmentation-lab/blob/main/examples/extensions/my_dataset.py).

## Before a Long Run

1. Run `verify-data` after copying or regenerating files.
2. Inspect train, valid, and test summaries and render representative overlays.
3. Exercise empty images, multiple classes, touching instances, and the largest image.
4. Force horizontal flip and resize in a synthetic test to verify boxes and masks stay aligned.
5. Run a real one-batch model update with `train --dry-run`.

A factory is trusted Python code. Keep downloaded annotations and images outside version control, publish their source/license and hashes, and never make a checkpoint depend on an undocumented local path.
