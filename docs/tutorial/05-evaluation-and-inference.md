# Evaluation and Inference

[中文](05-evaluation-and-inference.zh-CN.md) | [Documentation index](../README.md)

Evaluate the checkpoint selected on validation, then run single-image inference:

```bash
uv run instance-segment evaluate \
  --checkpoint artifacts/my-run/best.pt --split test --device cpu --plot
uv run instance-segment predict \
  --checkpoint artifacts/my-run/best.pt \
  --image data/raw/PennFudanPed/PNGImages/FudanPed00028.png \
  --output artifacts/my-run/prediction --device cpu
```

## Evaluation

The default metric floor is `0.0`, so all model outputs participate in confidence-ranked COCO AP. `--score-threshold` is separate and affects per-image matching, false-positive/false-negative counts, and overlays. `--mask-threshold` converts mask probabilities to binary masks. Every value is recorded in `evaluation.json`.

Evaluation traverses the split once. It writes bbox/mask AP and AR, per-class values, per-image error counts, and, with `--plot`, four ranked worst-case ground-truth/prediction pairs. The output directory is protected from accidental replacement; pass `--overwrite` deliberately.

## Prediction

Prediction is independent of the prepared dataset and accepts one RGB-convertible image. It writes:

```text
prediction/
  instances.json
  overlay.png
  masks/instance-001.png ...
```

`instances.json` contains the source path, thresholds, class ID/name, score, half-open `box_xyxy`, and relative mask path. A binary PNG is written for every retained instance, including the case of zero retained instances (the `masks/` directory remains present).

## Compare

```bash
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric mask_map
```

Use the comparison command only for runs with the same dataset identity, split hashes, class schema, metric protocol, and thresholds. See [Metrics](../reference/metrics.md) for interpretation and [CLI and outputs](../reference/cli-and-outputs.md) for the complete artifact layout.
