# 使用模型

使用 `instance-segment list-models` 和 `model-info`。外部模型工厂接收 `num_classes`、`weights`、`params`，返回 `nn.Module`，并遵守 torchvision 的训练/推理契约。`examples/extensions/my_segmenter.py` 调用内置 Mask R-CNN factory。
