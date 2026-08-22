# 模型选择

[English](choosing-models.md) | [文档导航](../README.zh-CN.md)

先根据要验证的问题选择模型：

| 场景 | 起点 | 原因 |
| --- | --- | --- |
| 校验数据、变换或 CPU 流程 | `maskrcnn_resnet50_fpn`、`weights: none`、`learning_minimal.yaml` | 与默认契约一致且不下载权重 |
| 复现已发布的 Penn-Fudan 基线 | `maskrcnn_resnet50_fpn`、`weights: coco_v1` | 这是 recorded protocol v2 路径 |
| 学习 backbone/anchor/ROI 设计 | `maskrcnn_mobilenet_v3_large` | 更轻的 backbone 和显式自定义组件 |
| 测试较新的 torchvision recipe | `maskrcnn_resnet50_fpn_v2` | 已安装 torchvision 支持时使用 |

写配置前先查看当前环境的模型清单：

```bash
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_mobilenet_v3_large
```

## 保持比较公平

比较模型时固定 dataset identity、label schema、split hash、seed、图片尺寸策略、batch size、优化器、学习率、epoch 预算和评估阈值。只修改 `model.name`、模型对应的权重/参数和 `run.name`。使用 `valid_mask_map` 选择模型，选定后再评估固定 test split。

当前清单包含：

- `maskrcnn_resnet50_fpn`：稳定的 ResNet-50 FPN 实现，支持 `none` 和 `coco_v1`。
- `maskrcnn_resnet50_fpn_v2`：较新的 torchvision ResNet-50 FPN recipe，取决于已安装 build。
- `maskrcnn_mobilenet_v3_large`：较轻的 backbone，支持 `none` 和 `imagenet_v2`。

模型构造参数放在 `model.params` 下，常用的是 `min_size` 和 `max_size`。不同模型族的权重名称不能想当然地复用，请先阅读[模型清单](../reference/model-zoo.zh-CN.md)和 `model-info`。

小数据集的点估计可能不稳定。每次比较都应同时记录图片/目标数量、训练耗时、峰值显存、最佳 epoch 和完整评估协议。已发布运行是教学用的有边界结果，不代表模型可泛化到无关领域。
