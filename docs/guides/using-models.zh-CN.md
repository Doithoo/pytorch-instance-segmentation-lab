# 使用模型

[English](using-models.md) | [文档导航](../README.zh-CN.md)

选择配置前先运行模型清单命令：

```bash
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_resnet50_fpn
```

`maskrcnn_resnet50_fpn` 是稳定的参考路径；v2 在可用时暴露较新的 torchvision recipe；`maskrcnn_mobilenet_v3_large` 展示更轻的 backbone、自定义 anchor 和显式 ROI pooling。

权重策略按模型区分：

- `none`：不下载权重。
- `coco_v1`：先加载完整 COCO Mask R-CNN 权重，再根据配置类别数替换两个 predictor。
- `imagenet_v2`：只初始化 MobileNet backbone。

外部 model factory 使用 `module.path:callable`，以 keyword-only 方式接收 `num_classes`、`weights` 和 `params`，并返回遵守 torchvision 训练/评估契约的 `nn.Module`。外部 factory 会执行可信 Python 代码，也可能下载额外资产。

完整训练前请检查预测头维度、有限的分项 loss、输出字段对齐、空预测、不同图片尺寸和 checkpoint 恢复。权重策略、构造参数、torchvision 版本以及下载/许可证信息都应记录在模型清单和实验元数据中。

公平比较见[模型选择](choosing-models.zh-CN.md)，扩展检查表见[添加模型](adding-models.zh-CN.md)。
