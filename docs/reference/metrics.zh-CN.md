# 指标

[English](metrics.md) | [文档导航](../README.zh-CN.md)

主选模指标是 `mask_map`：在 mask IoU `0.50:0.95` 多个阈值上平均的 COCO-style average precision。`bbox_map` 是对应的 box 指标。AP 使用 `torchmetrics.MeanAveragePrecision` 和 `pycocotools`，并保留置信度排序。

## 三个阈值

三个值有意承担不同职责：

| 字段 | 控制内容 | 默认值 |
| --- | --- | ---: |
| `evaluation_score_floor` / `--metric-score-floor` | 预测进入 AP 前的可选最低 score | `0.0` |
| `score_threshold` / `--score-threshold` | 逐图匹配、FP/FN 报告、预测展示和 overlay | `0.5` |
| `mask_threshold` / `--mask-threshold` | 将 mask 概率转为二值 mask | `0.5` |

协议 v2 将 metric floor 保持为 `0.0`。如果先把所有预测按 `0.5` 过滤再计算 AP，会改变 precision-recall 曲线，不能再称为标准的置信度排序 AP。

## 报告字段

`evaluation.json` 包含 mask/box 的 AP、AP50、AP75 和 AR@100，图片/目标/预测数量，类别名称，阈值，指标后端/协议，dataset identity 和 split hash。`per_class.csv` 按类别提供 mask 与 bbox AP。`per_image.csv` 提供目标/预测数量、分析阈值下的贪心 IoU 匹配、误报、漏报和低 IoU 数量。

传入 `--plot` 时，评估会保留严重程度最高的四张图片，并写入成对的 ground-truth/prediction overlay。这只是有界诊断视图，不改变指标。

## 如何解释

实例分割应以 mask AP 作为主指标，同时报告 bbox AP。box AP 高并不表示 mask 形状准确，因此两者不能互相替代。项目刻意不提供 pixel accuracy，因为大量背景像素会掩盖实例缺失。已发布 test 只有 17 张图片和 40 个目标；想做更广泛结论时，应配合多 seed 或按图片 bootstrap 区间。

运行比较要求 dataset identity、split hash、类别数、指标协议、score floor 和 mask threshold 兼容。`--allow-incompatible` 只用于检查差异，不能用来制造排名。
