# Evaluation and Inference

```bash
uv run instance-segment evaluate \
  --checkpoint artifacts/my-run/best.pt \
  --split test --plot

uv run instance-segment predict \
  --checkpoint artifacts/my-run/best.pt \
  --image path/to/image.png \
  --output artifacts/prediction
```

Evaluation defaults to `metric_score_floor=0.0`, preserving confidence-ranked COCO AP. `--score-threshold` affects only per-image error counts and overlays; `--metric-score-floor` is an explicit nonstandard pruning control and is recorded in JSON. Evaluation runs inference once and writes bbox/mask metrics, per-class values, per-image matches/errors, and four ranked worst cases.

Prediction writes `instances.json`, one binary PNG per retained instance, and an overlay. Verify checkpoint SHA-256 and load only trusted `.pt` files.

Compare completed compatible runs with:

```bash
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric mask_map
```
