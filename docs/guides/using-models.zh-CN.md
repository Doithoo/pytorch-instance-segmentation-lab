# 使用模型

选择配置前运行 `instance-segment list-models` 和 `model-info NAME`。ResNet50-FPN v1 是替代参考路径；v2 暴露较新的 torchvision recipe；MobileNetV3-Large 用较轻 backbone 展示显式自定义 anchor 与 ROI pooling。

权重策略按模型定义。`coco_v1` 加载完整 Mask R-CNN COCO 权重后替换两个 prediction head；`imagenet_v2` 只初始化 MobileNet backbone；`none` 不下载权重。

外部 model factory 使用 `module:callable`，以 keyword-only 方式接收 `num_classes`、`weights`、`params`，并返回符合 torchvision 训练/评估契约的 `nn.Module`。外部 factory 会执行可信 Python 代码。完整训练前应检查 head 维度、有限的分项 loss、输出字段对齐、空预测和 checkpoint 恢复。
