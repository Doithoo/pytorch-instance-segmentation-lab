# Evaluation and Inference

```bash
uv run instance-segment evaluate --checkpoint artifacts/my-run/best.pt --split test --plot
uv run instance-segment predict --checkpoint artifacts/my-run/best.pt --image path/to/image.png --output artifacts/prediction
```

Evaluation reports COCO-style bbox and mask AP. Prediction saves `instances.json`, one thresholded binary PNG per instance, and `overlay.png`; the JSON states both score and mask thresholds.
