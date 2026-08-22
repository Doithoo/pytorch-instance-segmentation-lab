# Model Zoo

[中文](model-zoo.zh-CN.md) | [Documentation index](../README.md)

| Name | Weight policies | Intended use |
| --- | --- | --- |
| `maskrcnn_resnet50_fpn` | `none`, `coco_v1` | Stable reference path and transfer learning |
| `maskrcnn_resnet50_fpn_v2` | `none`, `coco_v1` | Newer torchvision ResNet-50 FPN recipe when available |
| `maskrcnn_mobilenet_v3_large` | `none`, `imagenet_v2` | Lightweight educational backbone and custom components |

## Weight Behavior

`none` avoids weight downloads. `coco_v1` loads complete torchvision Mask R-CNN COCO weights before replacing box and mask predictors for the configured class count. It therefore initializes a transfer-learning backbone and heads, but the final head shape is project-specific. `imagenet_v2` initializes only the MobileNet backbone; it does not claim COCO-pretrained instance heads.

The MobileNet variant uses one feature map, custom anchors, explicit box ROI pooling, and explicit mask ROI pooling. It is intended to make those components inspectable, not to promise parity with the COCO-pretrained ResNet path.

`model.params` is forwarded to the selected constructor. The built-in notes currently cover `min_size` and `max_size`; run:

```bash
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_resnet50_fpn_v2
```

Before advertising a new model, add predictor head-shape tests, a real forward smoke test in its supported environment, checkpoint round-trip coverage, a documented weight/download policy, and a paired update to this catalog.
