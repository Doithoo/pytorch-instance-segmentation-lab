# PyTorch 实例分割

[![CI](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md)

可复现的 PyTorch 实例分割实现，包含 Penn-Fudan 与 COCO polygon/RLE 数据 provider、torchvision Mask R-CNN 模型、训练、评估、错误分析和 checkpoint 推理。

![Penn-Fudan 数据与实例](docs/recorded-run/assets/dataset-preview.png)

```text
下载 -> 准备 -> 校验 -> 检查 -> dry-run -> 训练 -> 评估 -> 对比/推理
```

## Kaggle 已完成运行

协议 v2 已在 [Kaggle kernel version 2](https://www.kaggle.com/code/yashowhoo/pytorch-instance-segmentation-lab-penn-fudan-gpu) 的 Tesla T4 上实际执行，使用仓库提交的按来源分层 manifests 和 20 个训练 epoch。

| 指标 | 结果 |
|---|---:|
| 最佳验证 mask AP（第 10 轮） | **0.766694** |
| Test mask AP / AP50 / AP75 | **0.756093** / 1.000000 / 0.855337 |
| Test bbox AP / AP50 / AP75 | **0.846439** / 1.000000 / 0.935175 |
| Test 图片 / 目标 | 17 / 40 |
| 训练 / 评估 / 总耗时 | 537.431s / 4.609s / 585.735s |

评估保留完整置信度排序（`metric_score_floor=0.0`）。固定 dataset identity 为 `64bfbd3d...b48d8`，最佳 checkpoint SHA-256 为 `1c28ed12...b3d57`。完整报告、provenance、可视化和模型卡见[训练记录](docs/recorded-run/README.zh-CN.md)。此前 score-filtered 的结果保留在 [`legacy-v1`](docs/recorded-run/legacy-v1/)，不能与协议 v2 直接比较。

## 项目范围

仓库提供：

- 独立 box、label 和二值 mask 的实例 target 契约。
- 固定按来源分层的 Penn-Fudan `136/17/17` manifests。
- 支持 polygon、RLE、多类别、crowd 和空图片的 COCO instance JSON 准备流程。
- ResNet50-FPN v1/v2 与 MobileNetV3-Large Mask R-CNN 模型构造器。
- 训练、验证集选模、选模后的 test 评估、checkpoint 恢复和单图推理。
- 机器可读指标、逐图错误报告、排序后的最差样本图和运行 provenance。

## 从全新克隆开始

需要 Python 3.10-3.12 和 [uv](https://docs.astral.sh/uv/)。以下命令从仓库根目录执行：

```bash
git clone https://github.com/Doithoo/pytorch-instance-segmentation-lab.git
cd pytorch-instance-segmentation-lab
uv sync --locked --extra dev
uv run instance-segment doctor --device auto
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run instance-segment inspect-data --split train
uv run python scripts/preview_dataset.py --output artifacts/dataset-preview.png
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

dry-run 会执行一次真实 optimizer update，但不写运行目录。接着阅读[教程](docs/tutorial/README.zh-CN.md)完成小规模训练、评估和预测。本地 CUDA 不是必需条件；完整 GPU 复现请使用 [Kaggle 指南](docs/guides/kaggle.zh-CN.md)。

## 文档

从[文档导航](docs/README.zh-CN.md)按目标进入：

- [教程](docs/tutorial/README.zh-CN.md)：基础、环境、数据、Mask R-CNN、训练、评估与推理。
- [概念](docs/concepts/code-tour.zh-CN.md)：target、模型、配置和代码流说明。
- [指南](docs/guides/choosing-models.zh-CN.md)：模型选择、自定义数据/模型、实验、Kaggle、排错和发布。
- [参考](docs/reference/cli-and-outputs.zh-CN.md)：CLI 命令、产物目录、配置、数据、指标、checkpoint 和模型。
- [训练记录](docs/recorded-run/README.zh-CN.md)：可审计的协议 v2 证据和模型卡。

可以使用 [`mkdocs.yml`](mkdocs.yml) 发布同一组页面。英文和中文页面按文件名成对维护，并由文档测试自动检查。

## 命令

- `instance-segment init-config --list`：列出已安装的配置模板。
- `instance-segment prepare-coco ...`：准备 COCO polygon/RLE 数据。
- `instance-segment list-models`：列出已注册的 Mask R-CNN 变体。
- `instance-segment evaluate`：生成指标、逐类 CSV、逐图错误和排序后的最差样本。
- `instance-segment compare-runs`：比较协议兼容的已完成运行。

配置与使用说明位于[文档导航](docs/README.zh-CN.md)、[指南](docs/guides/)、[参考](docs/reference/)和[架构决策](docs/architecture/)。

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
