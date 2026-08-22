# Environment

[中文](01-environment.zh-CN.md) | [Documentation index](../README.md)

The package supports Python 3.10, 3.11, and 3.12. [uv](https://docs.astral.sh/uv/) is the reference environment manager. Run all commands from the repository root.

## Install and Check

```bash
uv sync --locked --extra dev
uv run instance-segment --version
uv run instance-segment show-config
uv run instance-segment doctor --device auto
uv run python -m pytest
```

CPU is sufficient for tests, data verification, inspection, and the real small dry-run. A full Mask R-CNN run is much more practical on CUDA. `doctor` reports the selected device, availability, Torch version, and CUDA details; it does not install drivers or download weights.

## Device Policy

Use `--device cpu` when validating a workflow or when reproducibility matters more than speed. Use `--device cuda` only after `doctor --device cuda` succeeds. `auto` chooses an available supported accelerator according to the package policy. MPS is accepted by the CLI, but model and operator support depends on the local PyTorch/torchvision build.

Pretrained `coco_v1` and `imagenet_v2` policies may download weights on first construction. `weights: none` avoids that network dependency and is the default for local smoke work. The Kaggle reference runner explicitly enables Internet because it downloads the checksum-pinned dataset and initialization weights.

## Verify Before Training

```bash
uv run instance-segment doctor --device auto
uv run instance-segment show-config --config configs/learning_minimal.yaml
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

A successful dry-run prints image shapes, target counts, component losses, and `dry-run OK`. It performs a real optimizer update but intentionally does not create `artifacts/` output. Continue with the data tutorial once this check passes.
