# PyTorch Instance Segmentation Lab

[![CI](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Doithoo/pytorch-instance-segmentation-lab/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[中文](README.zh-CN.md)

A reproducible, teaching-oriented PyTorch instance-segmentation lab for independent masks, boxes, labels, training, COCO evaluation, error analysis, and checkpoint inference. It includes Penn-Fudan and COCO polygon/RLE providers plus torchvision Mask R-CNN models.

![Penn-Fudan dataset and instances](docs/recorded-run/assets/dataset-preview.png)

```text
download/prepare -> verify -> inspect -> dry-run -> train -> evaluate -> compare/predict
```

## Baseline status

Evaluation protocol v2 preserves the complete confidence ranking for standard COCO-style AP. Penn-Fudan manifests use a fixed source-stratified 136/17/17 split so both Fudan and Penn domains occur in every split.

The protocol-v2 20-epoch T4 run is complete. Epoch 10 reached validation mask AP `0.766694`; the fixed test split produced mask AP `0.756093` and bbox AP `0.846439` with the full confidence ranking. See the [auditable recorded run](docs/recorded-run/README.md).

The superseded score-filtered, lexicographic-split result remains under [`legacy-v1`](docs/recorded-run/legacy-v1/) and is not comparable with protocol v2.

## Local start

```bash
uv sync --locked --extra dev
uv run instance-segment doctor --device auto
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

A target is a list of independently labeled `boxes`, `labels`, and boolean `masks`, not one semantic class map. `best.pt` is selected only by validation `mask_map`; test evaluation remains a post-selection operation.

## Built-in workflows

- `instance-segment init-config --list`: discover wheel-installed configuration templates.
- `instance-segment prepare-coco ...`: prepare multiclass COCO polygon/RLE datasets, including empty images.
- `instance-segment list-models`: inspect ResNet50-FPN v1/v2 and MobileNetV3-Large Mask R-CNN variants.
- `instance-segment evaluate`: write COCO bbox/mask metrics, per-class CSV, per-image errors, and ranked worst cases in one inference pass.
- `instance-segment compare-runs`: compare only compatible dataset and metric protocols by default.

See the [tutorial](docs/tutorial/README.md), [guides](docs/guides/), [reference](docs/reference/), and [protocol-v2 decision](docs/architecture/0002-evaluation-and-splits.md).

## Kaggle full training

The generated runner embeds an exact source archive and fixed manifests, verifies a T4-or-newer GPU, downloads checksum-pinned data and weights, emits JSON heartbeats, and records full provenance. Follow the [Kaggle guide](docs/guides/kaggle.md). A complete protocol-v2 run must publish checkpoint and source hashes with its reports.

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
