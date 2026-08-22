# 模型列表

| 名称 | 权重策略 | 用途 |
|---|---|---|
| `maskrcnn_resnet50_fpn` | `none`、`coco_v1` | 稳定迁移学习与完整参考 runner |
| `maskrcnn_resnet50_fpn_v2` | `none`、`coco_v1` | 较新的 torchvision ResNet50-FPN recipe |
| `maskrcnn_mobilenet_v3_large` | `none`、`imagenet_v2` | 轻量、可拆解的自定义 backbone 教学模型 |

COCO Mask R-CNN 策略加载完整预训练权重，再按配置类别数替换 box 与 mask predictor。MobileNet 版本使用 ImageNet backbone、单特征图、自定义 anchor 和显式 box/mask ROI pooling，不声称具备 COCO 预训练实例 head。

`model.params` 会传给对应构造器，常见参数是 `min_size` 和 `max_size`。运行 `instance-segment model-info NAME` 查看说明。任何新公开模型都应有 head shape 测试、支持环境中的真实 forward smoke 和明确的权重策略文档。
