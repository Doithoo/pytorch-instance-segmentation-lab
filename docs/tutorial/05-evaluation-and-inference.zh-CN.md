# 评估与推理

```bash
uv run instance-segment evaluate --checkpoint artifacts/my-run/best.pt --split test --plot
uv run instance-segment predict --checkpoint artifacts/my-run/best.pt --image path/to/image.png --output artifacts/prediction
```

评估输出 COCO 风格 bbox/mask AP。推理保存 `instances.json`、每个实例的阈值化二值 PNG 和 `overlay.png`；JSON 明确记录 score 与 mask threshold。
