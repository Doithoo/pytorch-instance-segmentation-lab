# Training

[中文](04-training.zh-CN.md) | [Documentation index](../README.md)

Start with the small configuration, then remove limits and choose a GPU configuration only after the data contract passes:

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
uv run instance-segment train --config configs/reference_maskrcnn.yaml --device cuda
uv run instance-segment train --config configs/reference_maskrcnn.yaml --resume artifacts/reference-maskrcnn/last.pt
```

## What Training Records

A full run validates manifest hashes, resolves the configuration, records Python/Torch/device/git/lock provenance, and writes `metrics.csv`, `events.jsonl`, `best.pt`, and `last.pt`. Metrics include the five component losses, validation bbox/mask AP and AR, learning rate, epoch duration, and peak CUDA memory.

`best.pt` is selected only by validation `mask_map`. The final epoch is evaluated even when it falls outside `evaluate_every`. The training orchestrator never loads the test split; run `evaluate --split test` after model selection.

## Dry Run and Resume

`--dry-run` constructs the configured model and performs one real optimizer update without creating a run directory. It catches target dtype, image shape, predictor head, loss, and backward errors early.

Use `--resume artifacts/<run>/last.pt` to continue the same trajectory. Resume restores model, optimizer, scheduler, and RNG state and rejects changed immutable configuration, manifest hashes, or an inconsistent metrics tail. Only run name/output, total target epochs, device, and worker count may change. Use a new run without `--resume` for an intentional experiment fork.

Read [Configuration reference](../reference/config-reference.md), [Checkpoint schema](../reference/checkpoint-schema.md), and [Experiments](../guides/experiments.md) before changing a long run.
