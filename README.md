# PyTorch Instance Segmentation

[![CI](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[中文](README.zh-CN.md)

Reproducible PyTorch instance-segmentation implementation with Penn-Fudan and COCO polygon/RLE data providers, torchvision Mask R-CNN models, training, evaluation, error analysis, and checkpoint inference.

![Penn-Fudan dataset and instances](docs/recorded-run/assets/dataset-preview.png)

```text
download/prepare -> verify -> inspect -> dry-run -> train -> evaluate -> compare/predict
```

## Completed Kaggle Run

Protocol-v2 training was executed on [Kaggle kernel version 2](https://www.kaggle.com/code/yashowhoo/pytorch-instance-segmentation-lab-penn-fudan-gpu), using a Tesla T4, the committed source-stratified manifests, and 20 training epochs. The run completed at `2026-08-22T12:19:00Z`.

| Metric | Result |
|---|---:|
| Best validation mask AP (epoch 10) | **0.766694** |
| Test mask AP / AP50 / AP75 | **0.756093** / 1.000000 / 0.855337 |
| Test bbox AP / AP50 / AP75 | **0.846439** / 1.000000 / 0.935175 |
| Test images / targets | 17 / 40 |
| Training / evaluation / total | 537.431s / 4.609s / 585.735s |

The evaluation keeps the complete confidence ranking (`metric_score_floor=0.0`). The fixed dataset identity is `64bfbd3d...b48d8`; the best checkpoint SHA-256 is `1c28ed12...b3d57`. Full reports, provenance, visualizations, and the model card are in the [recorded run](docs/recorded-run/README.md). The previous score-filtered result is preserved under [`legacy-v1`](docs/recorded-run/legacy-v1/) and is not comparable with protocol v2.

## Scope

The repository provides:

- An instance target contract for independent boxes, labels, and binary masks.
- Penn-Fudan manifests with deterministic source-stratified `136/17/17` splits.
- COCO instance JSON preparation with polygon, RLE, multiclass, crowd, and empty-image support.
- ResNet50-FPN v1/v2 and MobileNetV3-Large Mask R-CNN model factories.
- Training, validation selection, post-selection test evaluation, checkpoint resume, and single-image inference.
- Machine-readable metrics, per-image error reports, ranked worst-case overlays, and run provenance.

## Local Reproduction

```bash
uv sync --locked --extra dev
uv run instance-segment doctor --device auto
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

The full GPU reproduction uses the [Kaggle runner](docs/guides/kaggle.md). The exact submitted protocol-v2 runner is [run_kaggle-v2.py](docs/recorded-run/kaggle/run_kaggle-v2.py).

## Commands

- `instance-segment init-config --list`: list installed configuration templates.
- `instance-segment prepare-coco ...`: prepare COCO polygon/RLE datasets.
- `instance-segment list-models`: list registered Mask R-CNN variants.
- `instance-segment evaluate`: write metrics, per-class CSV, per-image errors, and ranked worst cases.
- `instance-segment compare-runs`: compare compatible completed runs.

Detailed configuration and usage information is organized under [documentation](docs/README.md), [guides](docs/guides/), [reference](docs/reference/), and [architecture decisions](docs/architecture/).

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest --cov=instance_segmenter --cov-report=term-missing
uv run python scripts/build_kaggle_runner.py --check
uv run python -m build && uv run twine check dist/*
```

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [changelog](CHANGELOG.md) before contributing. PyTorch `.pt` checkpoints and external factories are trusted-code inputs; never load them from an unverified source.
