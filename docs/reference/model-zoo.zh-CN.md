# 模型清单

[English](model-zoo.md) | [文档导航](../README.zh-CN.md)

| 名称 | 权重策略 | 适用场景 |
| --- | --- | --- |
| `maskrcnn_resnet50_fpn` | `none`、`coco_v1` | 稳定参考路径和迁移学习 |
| `maskrcnn_resnet50_fpn_v2` | `none`、`coco_v1` | 可用时测试较新的 torchvision ResNet-50 FPN recipe |
| `maskrcnn_mobilenet_v3_large` | `none`、`imagenet_v2` | 轻量教学 backbone 和自定义组件 |

## 权重行为

`none` 不下载权重。`coco_v1` 会先加载完整的 torchvision COCO Mask R-CNN 权重，再为配置的类别数替换 box 和 mask predictor。因此它初始化了迁移学习 backbone 和 head，但最终 head shape 是项目自己的。`imagenet_v2` 只初始化 MobileNet backbone，不声称拥有 COCO 预训练的实例 head。

MobileNet 变体使用一个 feature map、自定义 anchor、显式 box ROI pooling 和显式 mask ROI pooling。它的目标是让这些组件可检查，不是保证和 COCO 预训练 ResNet 路径完全相同。

`model.params` 会转发给选定构造器。目前内置说明覆盖 `min_size` 和 `max_size`；可运行：

```bash
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_resnet50_fpn_v2
```

公开宣传新模型前，应添加预测头 shape 测试、支持环境中的真实 forward smoke、checkpoint round-trip、权重/下载策略说明，并同步更新本清单的中英文页面。
