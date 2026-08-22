# Mask R-CNN

[中文](03-maskrcnn.zh-CN.md) | [Documentation index](../README.md)

This project uses the torchvision detection API instead of reimplementing RPN, ROIAlign, NMS, or COCO AP. The learning value is in the data contract, model wiring, experiment protocol, and inspectable outputs.

## Forward Paths

During training, a batch is a list of float32 `CHW` images and a matching list of targets. Torchvision's Mask R-CNN returns a dictionary containing classifier, box-regression, mask, objectness, and RPN-box losses. The trainer sums the finite component losses and performs the optimizer update.

During evaluation, the model receives only images and returns one dictionary per image:

```text
boxes:  [N, 4] float coordinates
labels: [N]    class IDs
scores: [N]    confidence ranking
masks:  [N, 1, H, W] mask probabilities
```

The inference layer applies score filtering for display, thresholds mask probabilities, and writes independent instance files. The metric path can preserve all predictions so confidence-ranked AP remains meaningful.

## Model Construction

Run `list-models` to see available names and weight policies. When `num_classes` changes, the project replaces both the box predictor and mask predictor so the output head matches the persisted label schema. `model.params` is forwarded for model-specific constructor settings such as `min_size` and `max_size`.

Use [Choosing a model](../guides/choosing-models.md) for selection and [Model catalog](../reference/model-zoo.md) for weight behavior. The [Mask R-CNN flow](../concepts/maskrcnn-flow.md) follows tensors through backbone, RPN, ROI heads, and mask head.
