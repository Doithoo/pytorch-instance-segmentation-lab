# 环境

[English](01-environment.md) | [文档导航](../README.zh-CN.md)

项目支持 Python 3.10、3.11 和 3.12。[uv](https://docs.astral.sh/uv/) 是推荐的环境管理工具。以下命令都从仓库根目录执行。

## 安装与检查

```bash
uv sync --locked --extra dev
uv run instance-segment --version
uv run instance-segment show-config
uv run instance-segment doctor --device auto
uv run python -m pytest
```

测试、数据校验、检查、真实的小型 dry-run 都可以使用 CPU。完整 Mask R-CNN 训练更适合 CUDA。`doctor` 会报告选择的设备、可用性、Torch 版本和 CUDA 信息，不会安装驱动，也不会下载权重。

## 设备策略

验证流程或更重视速度以外的可复现性时使用 `--device cpu`。只有 `doctor --device cuda` 成功后才使用 `--device cuda`。`auto` 会按照包的策略选择可用的支持设备。CLI 接受 MPS，但具体模型和算子是否支持取决于本地 PyTorch/torchvision build。

`coco_v1` 和 `imagenet_v2` 权重策略可能在第一次构造模型时下载权重；本地 smoke 默认使用 `weights: none`，避免网络依赖。Kaggle 参考 runner 会显式打开 Internet，因为它需要下载带 checksum 的数据集和初始化权重。

## 训练前验证

```bash
uv run instance-segment doctor --device auto
uv run instance-segment show-config --config configs/learning_minimal.yaml
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

成功的 dry-run 会输出图片尺寸、target 数量、分项 loss 和 `dry-run OK`。它会执行真实 optimizer update，但刻意不创建 `artifacts/` 产物。通过后继续阅读数据教程。
