# Data and Instances

```bash
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run python scripts/preview_dataset.py --output artifacts/dataset-preview.png
```

The committed manifests order all 170 filenames deterministically into `136/17/17`. Source PNG and indexed mask SHA-256 values are checked before training. Each positive pixel ID in `PedMasks` becomes one boolean `[H, W]` instance mask.
