# 评估与推理

[English](05-evaluation-and-inference.md) | [文档导航](../README.zh-CN.md)

先评估 valid 选出的 checkpoint，再进行单图推理：

```bash
uv run instance-segment evaluate \
  --checkpoint artifacts/my-run/best.pt --split test --device cpu --plot
uv run instance-segment predict \
  --checkpoint artifacts/my-run/best.pt \
  --image data/raw/PennFudanPed/PNGImages/FudanPed00028.png \
  --output artifacts/my-run/prediction --device cpu
```

## 评估

默认指标 floor 是 `0.0`，因此所有模型输出都会参与按置信度排序的 COCO AP。`--score-threshold` 独立控制逐图匹配、误报/漏报数量和 overlay。`--mask-threshold` 将 mask 概率转换为二值 mask。所有数值都会记录到 `evaluation.json`。

评估只遍历 split 一次，会写入 bbox/mask AP 与 AR、逐类指标、逐图错误数量；传入 `--plot` 时还会写入四组按严重程度排序的标注/预测图片。输出目录默认防止误覆盖；替换时请显式传 `--overwrite`。

## 预测

预测不依赖 prepared dataset，接收一张可转换为 RGB 的图片，并写入：

```text
prediction/
  instances.json
  overlay.png
  masks/instance-001.png ...
```

`instances.json` 包含源路径、阈值、类别 ID/名称、score、半开区间 `box_xyxy` 和相对 mask 路径。每个保留实例都会生成一个二值 PNG；即使没有保留实例，`masks/` 目录仍会存在。

## 比较

```bash
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric mask_map
```

只比较 dataset identity、split hash、类别 schema、指标协议和阈值都一致的运行。指标解释见[指标参考](../reference/metrics.zh-CN.md)，完整产物结构见[CLI 与输出](../reference/cli-and-outputs.zh-CN.md)。
