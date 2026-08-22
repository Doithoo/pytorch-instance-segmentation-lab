# 评估与推理

```bash
uv run instance-segment evaluate \
  --checkpoint artifacts/my-run/best.pt \
  --split test --plot

uv run instance-segment predict \
  --checkpoint artifacts/my-run/best.pt \
  --image path/to/image.png \
  --output artifacts/prediction
```

评估默认使用 `metric_score_floor=0.0`，保留按置信度排序的 COCO AP。`--score-threshold` 只影响逐图错误数与 overlay；`--metric-score-floor` 是显式的非标准裁剪参数，并会写入 JSON。评估只执行一次推理，输出 bbox/mask 指标、逐类结果、逐图匹配/错误和 4 组排序后的最差样本。

推理输出 `instances.json`、每个保留实例的二值 PNG 和 overlay。请验证 checkpoint SHA-256，并且只加载可信 `.pt` 文件。

比较已完成且兼容的运行：

```bash
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric mask_map
```
