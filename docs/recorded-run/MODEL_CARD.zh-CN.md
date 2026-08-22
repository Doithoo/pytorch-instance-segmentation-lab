# 协议 v2 Mask R-CNN 模型卡

## 模型

- 架构：torchvision `maskrcnn_resnet50_fpn`
- 初始化：`MaskRCNN_ResNet50_FPN_Weights.COCO_V1`，再为 background/person 替换预测头
- 数据集：Penn-Fudan Pedestrian，按来源分层的 manifest，136 train / 17 valid / 17 test
- 训练：20 epoch、SGD、StepLR、水平翻转、CUDA AMP、seed 42
- 选模：第 10 轮，验证 mask AP 为 `0.766694`
- 环境：Kaggle Tesla T4，kernel version 2

## 评估

协议 v2 保留完整置信度排序（`metric_score_floor=0.0`），mask threshold 为 0.5。固定 test split 的结果：

| 指标 | 数值 |
|---|---:|
| Mask AP / AP50 / AP75 | 0.756093 / 1.000000 / 0.855337 |
| Box AP / AP50 / AP75 | 0.846439 / 1.000000 / 0.935175 |
| Mask AR@100 / Box AR@100 | 0.782500 / 0.872500 |

Test 只有 17 张图片和 40 个目标。这些数值是可复现参考结果，不代表生产环境泛化能力。

## 适用范围

checkpoint 适合学习、可复现实验、Penn-Fudan 推理、迁移学习演示和回归测试，预测单个前景类别 `person`。

它不适合安全关键决策、监控部署、人口统计性能结论，或未经新验证就直接用于无关领域。score 不是校准概率，小数据集也不足以支持广泛的公平性或鲁棒性结论。

## 产物与安全

- 本地忽略产物：`artifacts/protocol-v2-reference/best.pt`
- Kaggle 输出：kernel version 2 的 `artifacts/reference-maskrcnn/best.pt`
- SHA-256：`1c28ed12b3b9d380bb3888a99267c6d7694efcfd2a93b22867abbb2ccb6b3d57`
- 大小：334.8 MB

PyTorch `.pt` 文件基于 pickle，属于可信输入。加载前请验证 SHA-256，并且只加载可信 artifact。

```bash
uv run instance-segment predict \
  --checkpoint artifacts/protocol-v2-reference/best.pt \
  --image path/to/image.png \
  --output artifacts/prediction
```

来源和完整 provenance 见 [`run.yaml`](run.yaml)、[`config.yaml`](config.yaml)、[`environment.json`](environment.json)、[`metrics.csv`](metrics.csv) 和精确提交的 [`run_kaggle-v2.py`](kaggle/run_kaggle-v2.py)。代码采用 MIT License；Penn-Fudan 和上游预训练权重遵循各自条款。
