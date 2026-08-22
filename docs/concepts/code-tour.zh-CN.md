# 代码导览

[English](code-tour.md) | [文档导航](../README.zh-CN.md)

建议按数据流阅读实现。CLI 刻意保持轻量，只负责解析参数并调用可测试的包 API。

1. `src/instance_segmenter/config.py` 定义 dataclass 默认值、严格 YAML 合并、`--set` 覆盖和校验。
2. `src/instance_segmenter/data/manifest.py` 定义数据元信息、split hash，以及 Penn-Fudan 的准备和校验。
3. `src/instance_segmenter/data/coco.py` 解码 polygon/RLE 标注并构造 COCO provider。
4. `src/instance_segmenter/data/schema.py` 定义 `InstanceTarget`、`LabelSchema` 以及 dtype/shape 校验。
5. `src/instance_segmenter/data/transforms.py` 保持图片、mask、box 和 area 的几何关系一致。
6. `src/instance_segmenter/data/collate.py` 将不同尺寸图片和不同实例数量保留为 list。
7. `src/instance_segmenter/models/registry.py` 与 `models/torchvision_models.py` 暴露模型清单并替换预测头。
8. `src/instance_segmenter/training/trainer.py` 完成单 epoch 或真实 dry-run 更新；`training/train.py` 管理可复现训练循环。
9. `src/instance_segmenter/training/checkpoint.py` 校验并恢复可信训练状态。
10. `src/instance_segmenter/evaluation/metrics.py` 计算 COCO-style bbox/mask 指标；`evaluation/evaluate.py` 写报告和排序 overlay。
11. `src/instance_segmenter/inference/predictor.py` 将单个 checkpoint 和图片转换为 JSON、二值 mask 与 overlay。
12. `src/instance_segmenter/cli.py` 把这些 API 接到 `instance-segment` 命令。

## 扩展边界

数据 provider 必须返回满足 target 契约的 `(image, target)`，并且能通过 manifest 元数据定位。模型 factory 必须返回符合 torchvision detection 约定的 `torch.nn.Module`：

- 训练：`model(images, targets)` 返回有限的 loss 字典；
- 评估：`model(images)` 为每张图片返回一个 prediction 字典；
- 预测字段：`boxes`、`labels`、`scores` 和 `masks` 的第一维保持对齐。

以 `examples/extensions/` 下的示例作为扩展起点。使用显式的 `module.path:callable` 加载可信 factory，不要在 CLI 中隐藏任意 import。

## 产物边界

`src/` 放可复用行为；`scripts/` 放仓库和发布任务；`examples/` 展示契约，不另起一套 trainer；`docs/recorded-run/` 放证据和 provenance，不是包运行时使用的数据。新增功能时请保持这些边界，确保文档、测试和生成的 Kaggle runner 都可审计。
