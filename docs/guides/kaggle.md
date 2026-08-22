# Full Training on Kaggle

Kaggle GPU is the required reference-training path. The runner embeds the exact source archive and the committed 136/17/17 manifests. It needs no Kaggle Dataset attachment or local CUDA.

## Submit

```bash
uv tool install kaggle
kaggle auth login
uv run python scripts/build_kaggle_runner.py --check
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-instance-segmentation-lab-penn-fudan-gpu
```

Change only `id` in `kernel-metadata.json`. Keep Internet enabled: the task downloads the official Penn-Fudan archive and COCO weights. Request T4 or newer; P100 may be incompatible with the installed PyTorch CUDA kernels. The task uses `cuda:0` even if Kaggle displays two GPUs.

## What the runner does

It reports JSON `started`, `running`, and `completed` events, with a 60 second heartbeat during slow phases. Its path is: GPU preflight, checksum download, manifest verification, preview, real Mask R-CNN dry-run, full 20 epoch training, one test evaluation of `best.pt`, one test-image prediction, and a summary.

An incomplete run is not a reference result. A completed run must report `completed_epochs: 20`, fixed split counts, a validation-selected `best_epoch`, and test metrics.

## Download results

```bash
kaggle kernels output <username>/pytorch-instance-segmentation-lab-penn-fudan-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

Inspect `reference-maskrcnn/best.pt`, `last.pt`, `metrics.csv`, `evaluation/`, predictions, `dataset-preview.png`, and `kaggle-run-summary.json`. On failure, inspect `kaggle-run-failure.json`; do not call its partial result a completed training record.
