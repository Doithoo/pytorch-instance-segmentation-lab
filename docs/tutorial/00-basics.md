# Instance Segmentation Basics

[中文](00-basics.zh-CN.md) | [Documentation index](../README.md)

Object detection returns a variable-length list of boxes. Semantic segmentation returns one class per pixel. Instance segmentation combines both ideas: every object remains an independent record with its class, box, binary mask, and confidence score.

## Why Instances Matter

Two people can touch or overlap while still being two training targets. A semantic foreground mask cannot represent that distinction without an additional instance-ID convention. This project reads instance IDs from masks, converts each positive ID into one object, and keeps the records separate through transforms, batching, training, metrics, and visualization.

## The Project Contract

For one image of height `H` and width `W`, the provider returns:

```text
image:  float32 [3, H, W], values in [0, 1]
target:
  boxes:   float32 [N, 4] in half-open xyxy order
  labels:  int64   [N]
  masks:   bool    [N, H, W]
  image_id:int64   [1]
  area:    float32 [N] pixel area
  iscrowd: int64   [N]
```

All fields with an `N` dimension describe the same instances and must stay aligned. `labels` contains foreground IDs only; label `0` is reserved for background in the shared schema. An image may have `N=0` in the COCO provider.

## End-to-End Shape

```text
image + instance-ID mask
        |
        v
independent boxes, labels, bool masks
        |
        v
Mask R-CNN losses during training
        |
        v
boxes, labels, scores, probability masks during inference
        |
        v
COCO AP, error tables, thresholded PNG masks, overlay
```

Start with the executable examples in the [examples directory on GitHub](https://github.com/Doithoo/pytorch-instance-segmentation-lab/tree/main/examples). Then continue to [Data and instances](02-data-and-instances.md) to see how the Penn-Fudan files become this target.
