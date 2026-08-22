# Troubleshooting

- `CUDA was requested`: run CPU dry-run or request T4/newer in Kaggle.
- manifest hash mismatch: rerun download/prepare; do not edit CSV rows.
- no predictions: inspect score threshold separately from mask threshold.
- mask/box drift: verify nearest-neighbor mask resize and box recomputation.
- checkpoint schema mismatch: use a checkpoint generated from the same label schema.
