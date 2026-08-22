# Kaggle 参考训练记录

[English](README.md) | [Kaggle 流程](../guides/kaggle.zh-CN.md) | [参考配置](../../configs/reference_maskrcnn.yaml)

强制参考运行已在 [Kaggle Tesla T4](https://www.kaggle.com/code/yashowhoo/pytorch-instance-segmentation-lab-penn-fudan-gpu) 成功完成。它使用提交的 Penn-Fudan `136/17/17` 划分、COCO 初始化的 torchvision Mask R-CNN ResNet-50 FPN、CUDA AMP 和全部 20 个 epoch。第 13 轮由验证集 `mask_map` 选出，随后仅用 `best.pt` 在留出的 test split 评估一次。

| 项目 | 结果 |
|---|---:|
| 最佳验证 mask AP | 0.795231 |
| 测试 mask AP / AP50 / AP75 | **0.791271** / 1.000000 / 0.966054 |
| 测试 bbox AP / AP50 / AP75 | **0.891579** / 1.000000 / 1.000000 |
| 测试 mask AR@100 / bbox AR@100 | 0.811429 / 0.911429 |
| 测试图片 / 目标 / 预测 | 17 / 35 / 37 |
| 训练 / test 评估 | 525.500s / 6.251s |
| Kaggle 任务总时间 | 570.792s |

![Penn-Fudan 数据集预览](assets/dataset-preview.png)

## 可审计产物

- [`config.yaml`](config.yaml)：Kaggle 实际路径、`cuda`、AMP、两个 worker 和完整 20 epoch。
- [`run.yaml`](run.yaml)：环境、split/source identity、耗时和 checkpoint hash。
- [`metrics.csv`](metrics.csv)：全部 20 轮训练/验证指标。
- [`kaggle-run-summary.json`](kaggle-run-summary.json)：runner 写入的未舍入最终数值。
- [`evaluation/evaluation.json`](evaluation/evaluation.json)、[`per_class.csv`](evaluation/per_class.csv)、[`per_image.csv`](evaluation/per_image.csv)：最终 test 结果。
- [`evaluation/visualizations/`](evaluation/visualizations/)：4 组 test 图片标注/预测对照。
- [`kaggle/run_kaggle-v1.py`](kaggle/run_kaggle-v1.py)：本次记录实际提交的生成 runner，内嵌 archive SHA-256 为 `c96eef...71d91`。
- [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py)：当前重新生成的 runner，下一次提交前由 CI 检查新鲜度。

335 MB 的 `best.pt`、`last.pt`、原始数据和完整 prediction cache 不进入仓库，应从 Kaggle kernel output 下载。被评估的 `best.pt` SHA-256 为 `af68b78d28b7b063e6932adc307a4db5f61f506409e88b9871447d45893bee6b`。
