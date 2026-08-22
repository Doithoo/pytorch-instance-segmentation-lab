# Instance Target

[中文](instance-target.zh-CN.md) | [Documentation index](../README.md)

`InstanceTarget` is the boundary between every dataset provider and the torchvision model. It is a typed dictionary with exactly six fields:

```python
{
    "boxes": float32[N, 4],
    "labels": int64[N],
    "masks": bool[N, H, W],
    "image_id": int64[1],
    "area": float32[N],
    "iscrowd": int64[N],
}
```

Boxes use half-open `xyxy`: `[x1, y1, x2, y2]`, with `x1 < x2` and `y1 < y2`. `area` is the number of true pixels in each binary mask. All fields describing instances share the same `N`; geometric transforms must update every aligned field.

The label schema is contiguous with background at ID `0`. Providers may have different source category IDs, but preparation maps them into the model schema and persists the mapping. `num_classes` must equal the schema count, including background.

An empty target is valid when the provider supports empty images. It still has the same fields and dtypes, with zero-length instance tensors and a one-element `image_id`. This is important for both model smoke tests and COCO datasets containing negative images.

The collate function returns `list[image]` and `list[target]`, not a padded batch tensor. This matches torchvision detection models and avoids inventing fake instances for images with different sizes or counts.

Use [the examples directory on GitHub](https://github.com/Doithoo/pytorch-instance-segmentation-lab/tree/main/examples) and the schema validator in `src/instance_segmenter/data/schema.py` when implementing a provider.
