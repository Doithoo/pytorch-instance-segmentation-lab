# Training

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
uv run instance-segment train --config configs/reference_maskrcnn.yaml --device cuda
uv run instance-segment train --config configs/reference_maskrcnn.yaml --resume artifacts/reference-maskrcnn/last.pt
```

A dry-run performs one real optimizer update but writes no run directory. Full training validates split hashes, records resolved config/environment/git/lock provenance, and writes `metrics.csv`, `events.jsonl`, `best.pt`, and `last.pt`. Metrics include component losses, validation bbox/mask AP and AR, learning rate, epoch duration, and peak CUDA memory.

`best.pt` is selected by validation mask AP. The final epoch is always evaluated. Test is never loaded by the training orchestrator. Resume restores optimizer, scheduler, and RNG state and rejects changed immutable configuration, split hashes, or a mismatched metrics tail.

Use the [Kaggle guide](../guides/kaggle.md) to reproduce the completed protocol-v2 reference run.
