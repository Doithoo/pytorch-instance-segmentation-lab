# PyTorch Instance Segmentation Lab

[中文](README.zh-CN.md)

Learn reproducible instance segmentation with Penn-Fudan Pedestrian and torchvision Mask R-CNN. The local path validates data and executes an actual CPU dry-run; the required full reference run uses a self-contained Kaggle GPU runner.

```text
download -> prepare -> inspect -> dry-run -> full Kaggle train -> evaluate -> predict
```

## Recorded Kaggle run

The required 20 epoch T4 reference run completed successfully. It selected epoch 13 by validation mask AP and evaluated the fixed 17-image test split once afterwards.

| Metric | Result |
|---|---:|
| Validation mask AP at best epoch | 0.795231 |
| Test mask AP / AP50 / AP75 | **0.791271** / 1.000000 / 0.966054 |
| Test bbox AP / AP50 / AP75 | **0.891579** / 1.000000 / 1.000000 |
| Test images / targets / predictions | 17 / 35 / 37 |
| Kaggle task time | 570.792s |

The [recorded run](docs/recorded-run/README.md) contains the Kaggle URL, resolved config, 20 epoch metrics, test reports, source/manifest/checkpoint hashes, and real overlays.

## Local start

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

The instance target is a list of independently labeled `boxes`, `labels`, and `masks`, not one semantic class map. `best.pt` is selected by validation `mask_map`; test evaluation runs only after selection.

## Kaggle full training

The release baseline is 20 epochs on the fixed 136/17/17 split using COCO-initialized Mask R-CNN on a Kaggle T4 or newer compatible GPU. The generated runner embeds an exact source archive, downloads data and COCO weights with Internet enabled, emits heartbeats, and writes only `artifacts` as useful output. Follow the [Kaggle guide](docs/guides/kaggle.md).

## Learning and development

Read [tutorials](docs/tutorial/README.md), [reference](docs/reference/), and the [architecture specification](docs/architecture/0001-instance-segmentation-lab.md). Before changes run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest
uv run python scripts/build_kaggle_runner.py --check
```
