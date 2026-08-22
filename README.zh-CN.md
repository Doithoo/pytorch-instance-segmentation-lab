# PyTorch 实例分割学习实验室

[![CI](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md)

这是一个面向学习、强调可复现性的 PyTorch 实例分割实验室，覆盖独立 mask/box/label、训练、COCO 评估、错误分析和 checkpoint 推理。内置 Penn-Fudan、COCO polygon/RLE 数据 provider，以及 torchvision Mask R-CNN 模型。

![Penn-Fudan 数据与实例](docs/recorded-run/assets/dataset-preview.png)

```text
下载/准备 -> 校验 -> 检查 -> dry-run -> 训练 -> 评估 -> 对比/推理
```

## 基线状态

评估协议 v2 保留完整预测置信度排序，以计算标准 COCO 风格 AP。Penn-Fudan 使用固定、按来源分层的 136/17/17 划分，Fudan 与 Penn 两个域都会进入 train、valid 和 test。

协议 v2 的 20 epoch T4 训练已完成。第 10 轮验证 mask AP 为 `0.766694`；保留完整置信度排序后，固定 test split 的 mask AP 为 `0.756093`、bbox AP 为 `0.846439`。完整信息见[可审计训练记录](docs/recorded-run/README.zh-CN.md)。

已被替代的 score-filtered、连续切分结果保留在 [`legacy-v1`](docs/recorded-run/legacy-v1/)，不能与协议 v2 直接比较。

## 本地开始

```bash
uv sync --locked --extra dev
uv run instance-segment doctor --device auto
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

实例 target 是独立的 `boxes`、`labels` 和 bool `masks` 列表，不是单张语义类别图。`best.pt` 只由验证集 `mask_map` 选择，test 始终在选模结束后单独评估。

## 内置工作流

- `instance-segment init-config --list`：发现 wheel 中安装的配置模板。
- `instance-segment prepare-coco ...`：准备多类别 COCO polygon/RLE 数据，也支持无实例图片。
- `instance-segment list-models`：查看 ResNet50-FPN v1/v2 与 MobileNetV3-Large Mask R-CNN。
- `instance-segment evaluate`：一次推理生成 bbox/mask 指标、逐类 CSV、逐图错误与最差样本图。
- `instance-segment compare-runs`：默认只比较数据和评估协议兼容的运行。

进一步阅读[教程](docs/tutorial/README.zh-CN.md)、[指南](docs/guides/)、[参考](docs/reference/)和[协议 v2 决策](docs/architecture/0002-evaluation-and-splits.zh-CN.md)。

## Kaggle 完整训练

生成的 runner 内嵌精确源码归档和固定 manifests，检查 T4 或更新 GPU，下载带 checksum 的数据与权重，输出 JSON heartbeat，并记录完整 provenance。操作见 [Kaggle 指南](docs/guides/kaggle.zh-CN.md)。完整协议 v2 运行必须同时发布 checkpoint、源码 hash 和评估报告。

## 开发

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest --cov=instance_segmenter --cov-report=term-missing
uv run python scripts/build_kaggle_runner.py --check
uv run python -m build && uv run twine check dist/*
```

贡献前阅读 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)、[SECURITY.md](SECURITY.md)与[变更日志](CHANGELOG.md)。PyTorch `.pt` checkpoint 和外部 factory 都属于可信代码输入，不要加载来源未经验证的文件。
