# PyTorch 实例分割学习实验室

[English](README.md)

使用 Penn-Fudan Pedestrian 与 torchvision Mask R-CNN 学习可复现实例分割。本地路径完成数据校验和真实 CPU dry-run；强制的完整参考训练使用自包含 Kaggle GPU runner。

```text
下载 -> 准备 -> 检查 -> dry-run -> Kaggle 完整训练 -> 评估 -> 推理
```

## 已完成的 Kaggle 训练

强制的 T4 参考运行已成功完成 20 个 epoch。第 13 轮由验证 mask AP 选出，之后只在固定的 17 张 test 图片上评估一次。

| 指标 | 结果 |
|---|---:|
| 最佳验证 mask AP | 0.795231 |
| 测试 mask AP / AP50 / AP75 | **0.791271** / 1.000000 / 0.966054 |
| 测试 bbox AP / AP50 / AP75 | **0.891579** / 1.000000 / 1.000000 |
| 测试图片 / 目标 / 预测 | 17 / 35 / 37 |
| Kaggle 任务时间 | 570.792s |

[训练记录](docs/recorded-run/README.zh-CN.md)保留 Kaggle URL、解析配置、20 轮指标、test 报告、source/manifest/checkpoint hash 和真实 overlay。

## 本地开始

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

实例 target 是独立的 `boxes`、`labels`、`masks` 列表，而不是单张语义类别图。`best.pt` 只由验证集 `mask_map` 选择，test 只在选模结束后评估。

## Kaggle 完整训练

发布基线是在固定 136/17/17 划分上，使用 COCO 初始化的 Mask R-CNN，在 Kaggle T4 或更新兼容 GPU 完成 20 epoch。生成的 runner 内嵌精确源码归档，在开启 Internet 后下载数据和 COCO 权重，输出心跳日志，并把有用结果保存在 `artifacts`。步骤见 [Kaggle 指南](docs/guides/kaggle.zh-CN.md)。

## 学习与开发

阅读[教程](docs/tutorial/README.zh-CN.md)、[参考](docs/reference/)和[实施规格](docs/architecture/0001-instance-segmentation-lab.zh-CN.md)。改动前运行：

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest
uv run python scripts/build_kaggle_runner.py --check
```
