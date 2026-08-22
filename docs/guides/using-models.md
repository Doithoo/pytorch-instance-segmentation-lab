# Using Models

[中文](using-models.zh-CN.md) | [Documentation index](../README.md)

Run the catalog commands before selecting a configuration:

```bash
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_resnet50_fpn
```

`maskrcnn_resnet50_fpn` is the stable reference path. The v2 variant exposes a newer torchvision recipe when available. `maskrcnn_mobilenet_v3_large` demonstrates a lighter backbone, custom anchors, and explicit ROI pooling.

Weight policies are model-specific:

- `none` performs no weight download.
- `coco_v1` loads complete COCO Mask R-CNN weights before replacing both predictors for the configured class count.
- `imagenet_v2` initializes only the MobileNet backbone.

An external model factory uses `module.path:callable`, receives keyword-only `num_classes`, `weights`, and `params`, and returns an `nn.Module` following the torchvision training/evaluation contract. External factories execute trusted Python code and may download additional assets.

Before a full run, validate predictor head dimensions, finite component losses, output-field alignment, empty predictions, different image sizes, and checkpoint restoration. Keep the weight policy, constructor parameters, torchvision version, and any download/license information in the model catalog and experiment metadata.

See [Choosing a model](choosing-models.md) for fair comparisons and [Adding a model](adding-models.md) for the extension checklist.
