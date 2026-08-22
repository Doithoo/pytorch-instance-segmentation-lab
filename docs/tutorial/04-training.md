# Training

Check one actual update locally first:

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

A full run records resolved config, environment, manifest hashes, epoch losses, validation metrics, `best.pt`, and `last.pt`. `best.pt` is selected only by validation `mask_map`; the test split is never used by `train`.

For the required 20-epoch GPU reference workflow, follow the [Kaggle guide](../guides/kaggle.md).
