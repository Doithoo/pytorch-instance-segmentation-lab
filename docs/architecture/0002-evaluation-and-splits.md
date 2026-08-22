# ADR 0002: Evaluation Protocol and Source-Stratified Splits

Status: Accepted

## Context

The initial release filtered predictions at score 0.5 before `MeanAveragePrecision`. That changes the confidence-ranked precision-recall curve and is not directly comparable with standard COCO AP. Its lexicographic Penn-Fudan split also put every Fudan image in train while valid and test contained only consecutive Penn images.

## Decision

Protocol v2 uses separate thresholds:

- `training.evaluation_score_floor` defaults to `0.0` and controls only an optional metric input floor.
- `training.score_threshold` defaults to `0.5` and controls prediction display and per-image error analysis.
- `training.mask_threshold` binarizes mask logits for both paths.

Penn-Fudan manifests use `source-stratified-sha256-v2`, seed 42. Each Fudan/Penn source group is deterministically hash-ordered and allocated into the fixed 136/17/17 totals. The committed split composition is 59/77, 7/10, and 8/9 Fudan/Penn for train, valid, and test.

Evaluation performs one model traversal. The same CPU predictions feed bbox/mask metrics, per-image reports, and a bounded set of ranked worst-case overlays.

Checkpoints must match current manifest hashes during resume and dataset-backed evaluation. Run comparison rejects different split hashes, metric protocol, score floor, mask threshold, or class count unless explicitly overridden.

## Consequences

The version 0.1.0 T4 metrics remain useful as a historical execution record but are marked legacy and cannot be relabeled as protocol-v2 results. Kaggle kernel version 2 subsequently completed the required replacement run; its reports and exact submitted runner are stored under `docs/recorded-run`.
