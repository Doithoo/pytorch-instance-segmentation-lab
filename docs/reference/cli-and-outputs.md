# CLI and Outputs

[中文](cli-and-outputs.zh-CN.md) | [Documentation index](../README.md)

Run commands from the repository root. `uv run` makes sure the command uses the project environment.

## Command Map

| Command | Purpose | Main result |
| --- | --- | --- |
| `show-config` | Merge defaults, YAML, and CLI overrides | Resolved YAML on stdout |
| `init-config` | List or copy installed templates | A YAML file |
| `doctor` | Inspect CPU/CUDA/MPS availability | Device report on stdout |
| `prepare-data` | Prepare Penn-Fudan manifests | `dataset.yaml`, `source.yaml`, and split CSVs |
| `prepare-coco` | Prepare manifests from three COCO instance JSON files | The same manifest contract |
| `verify-data` | Check source files, dimensions, and hashes | Verification summary |
| `inspect-data` | Summarize one prepared split | YAML summary on stdout |
| `list-datasets` | List registered data providers | Names and descriptions |
| `list-models` | List registered models and weights | Model catalog on stdout |
| `model-info NAME` | Show model notes and parameters | Teaching metadata |
| `train --dry-run` | Perform one real update | Diagnostics on stdout, no run directory |
| `train` | Train and validate | A run directory |
| `evaluate` | Evaluate one checkpoint on one split | `evaluation/` reports |
| `predict` | Predict instances for one image | JSON, masks, and overlay |
| `compare-runs` | Rank compatible runs | Tab-separated ranking on stdout |

Inspect the complete parser contract with `uv run instance-segment --help` and the command-specific `--help` output. Threshold options accept finite values from `0` through `1`.

## First Commands

```bash
uv run instance-segment show-config --config configs/learning_minimal.yaml
uv run instance-segment doctor --device auto
uv run instance-segment list-datasets
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_resnet50_fpn
```

Configuration commands accept repeated `--set KEY VALUE` arguments. Precedence is defaults, YAML, then CLI. For example:

```bash
uv run instance-segment train --config configs/learning_minimal.yaml \
  --set run.name first-cpu-run --set data.train_limit 2 --device cpu
```

## Run Layout

A normal training run writes `artifacts/<run.name>/`:

```text
config.yaml              resolved configuration
manifest-hashes.yaml     hashes used by the run
environment.json         Python, Torch, device, git, and lock provenance
events.jsonl             lifecycle and epoch events
metrics.csv              losses, validation metrics, and runtime values
best.pt                  checkpoint selected by validation mask_map
last.pt                  checkpoint from the final completed epoch
evaluation/              optional reports created by evaluate
```

`train --dry-run` returns image shapes, target counts, component losses, and `dry-run OK`; it intentionally creates no run directory. A normal run refuses to reuse an existing directory unless it is a valid `--resume` operation.

## Evaluation Output

By default, evaluating `artifacts/run/best.pt` writes beside the checkpoint:

```text
artifacts/run/evaluation/
  evaluation.json
  per_class.csv
  per_image.csv
  visualizations/          only with --plot
    worst-*-ground-truth.png
    worst-*-prediction.png
```

The JSON records the split, metric backend and protocol, all thresholds, class names, dataset identity, split hashes, counts, and metric values. `--overwrite` is required when replacing an existing report directory.

```bash
uv run instance-segment evaluate --checkpoint artifacts/first-cpu-run/best.pt \
  --split test --device cpu --output-dir artifacts/first-cpu-run/test-evaluation --plot
```

## Prediction Output

`predict` requires an image and an output directory. It writes one thresholded grayscale PNG per retained instance, an `instances.json` index, and `overlay.png`:

```bash
uv run instance-segment predict --checkpoint artifacts/first-cpu-run/best.pt \
  --image data/raw/PennFudanPed/PNGImages/FudanPed00028.png \
  --output artifacts/first-cpu-run/prediction --device cpu
```

`instances.json` stores the source image, thresholds, class ID/name, confidence score, half-open `box_xyxy`, and relative mask path. Prediction is single-image inference and does not verify dataset manifests; verify the checkpoint hash before loading it.

## Run Comparison

Use `valid_mask_map` to compare training runs selected on validation, or `mask_map` to compare runs with compatible evaluation reports:

```bash
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric valid_mask_map
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric mask_map --allow-incompatible
```

Comparison rejects mismatched dataset identity, split hashes, class count, metric protocol, score floor, or mask threshold. `--allow-incompatible` is for diagnosis and should not be used for a published ranking.
