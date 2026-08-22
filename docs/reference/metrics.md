# Metrics

`mask_map` is COCO-style AP averaged over mask IoU 0.50:0.95 and selects `best.pt`; `bbox_map` is the box equivalent. AP uses prediction confidence ranking, so protocol v2 sends all model outputs at or above `training.evaluation_score_floor` (default `0.0`) to `torchmetrics.MeanAveragePrecision`.

`training.score_threshold` is separate. It controls prediction files, overlays, and per-image match/error counts, not AP. `training.mask_threshold` converts mask probabilities to binary masks.

Reports include AP/AP50/AP75, AR@100, per-class mask/bbox AP, image/target/prediction counts, and error counts. An evaluation JSON records the metric backend, protocol, floor, thresholds, dataset identity, and split hashes. Compare runs only when these fields agree.

Pixel accuracy is intentionally omitted because dominant background pixels can hide failed instances. For small test sets, publish image-level bootstrap confidence intervals or repeated-seed results alongside point estimates.
