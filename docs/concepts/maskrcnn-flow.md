# Mask R-CNN Flow

[中文](maskrcnn-flow.zh-CN.md) | [Documentation index](../README.md)

The runtime path is a list-oriented detection pipeline:

```text
list[float32 CHW images]
        |
        v
Torchvision transform and backbone
        |
        v
ResNet-50 or MobileNetV3 features
        |
        v
FPN (where configured) -> RPN proposals
        |
        v
ROI box head: class and box refinement
        |
        v
ROI mask head: one mask probability map per retained instance
```

The RPN and ROI heads operate on proposals, so the number of output instances is not fixed by the batch. During training, target boxes, labels, masks, areas, and crowd flags supervise the appropriate branches. During evaluation, only the image list is passed and postprocessing returns aligned fields.

The project changes the final box and mask predictors for the configured `num_classes`. This is why a COCO-pretrained backbone/head cannot be loaded unchanged for a two-class Penn-Fudan task. Weight policies describe exactly which upstream components are initialized; see the [model catalog](../reference/model-zoo.md).

Resize policy belongs to the model constructor for torchvision models (`min_size`/`max_size`) and to the dataset config when the input target itself must be resized. Keep these choices consistent when comparing runs. The inference output then thresholds mask probabilities and writes one artifact per instance rather than a single semantic label image.
