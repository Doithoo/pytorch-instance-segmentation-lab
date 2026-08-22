# Metrics

[中文](metrics.zh-CN.md) | [Documentation index](../README.md)

The primary selection metric is `mask_map`: COCO-style average precision over mask IoU thresholds `0.50:0.95`. `bbox_map` is the corresponding box metric. AP is computed by `torchmetrics.MeanAveragePrecision` with `pycocotools` and uses confidence ranking.

## Thresholds

Three values have intentionally separate jobs:

| Field | Controls | Default |
| --- | --- | ---: |
| `evaluation_score_floor` / `--metric-score-floor` | Optional lower score floor before predictions enter AP | `0.0` |
| `score_threshold` / `--score-threshold` | Per-image matching, FP/FN reports, prediction display, and overlays | `0.5` |
| `mask_threshold` / `--mask-threshold` | Converts mask probabilities to binary masks | `0.5` |

Protocol v2 keeps the metric floor at `0.0`. Filtering all predictions at `0.5` before AP changes the precision-recall curve and must not be presented as standard confidence-ranked AP.

## Report Fields

`evaluation.json` includes AP/AP50/AP75 and AR@100 for masks and boxes, image/target/prediction counts, class names, threshold values, metric backend/protocol, dataset identity, and split hashes. `per_class.csv` contains mask and bbox AP by class. `per_image.csv` contains target/prediction counts, greedy IoU matches at the analysis threshold, false positives, false negatives, and low-IoU cases.

With `--plot`, evaluation keeps four highest-severity samples and writes paired ground-truth/prediction overlays. This is a bounded diagnostic view; it does not change the metrics.

## Interpretation

Mask AP should be the headline metric for instance segmentation. Bbox AP can be high while masks are poorly shaped, so report both. Pixel accuracy is intentionally absent because a dominant background can conceal missing instances. The published test split contains only 17 images and 40 targets; pair point estimates with repeated seeds or image-level bootstrap intervals when making a broader claim.

Run comparison requires compatible dataset identity, split hashes, class count, metric protocol, score floor, and mask threshold. Use `--allow-incompatible` only to inspect differences, never to manufacture a ranking.
