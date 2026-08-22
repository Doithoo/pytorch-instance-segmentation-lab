# Full Training on Kaggle

The generated runner is the reproducible GPU path for protocol v2. It embeds the exact package source, reference config, and source-stratified manifests; no Kaggle Dataset mount is required.

```bash
uv run python scripts/build_kaggle_runner.py
cd docs/recorded-run/kaggle
kaggle kernels push -p .
kaggle kernels status Doithoo/pytorch-instance-segmentation-lab-penn-fudan-gpu
kaggle kernels output Doithoo/pytorch-instance-segmentation-lab-penn-fudan-gpu -p output
```

Change only `id` in `kernel-metadata.json`. Keep Internet enabled because the task downloads the checksum-pinned Penn-Fudan archive, COCO initialization weights, and missing metric dependencies. Request a T4 or newer compatible NVIDIA GPU.

The runner emits JSON started/running/completed events and 60-second heartbeats. It performs GPU preflight, source download and verification, preview, real dry-run, 20-epoch train/validation, best-checkpoint selection, one test evaluation, prediction, summary writing, and cleanup of runtime data/project directories so future kernel outputs remain artifact-focused.

Kernel version 2 predates that final cleanup and therefore exposes raw runtime directories in its downloadable output. To retrieve only its useful files, use `kaggle kernels output ... --file-pattern '^artifacts/'`.

A protocol-v2 result is complete only when it reports 20 epochs, the current dataset identity `64bfbd3d...`, metric score floor `0.0`, a finite best epoch, test reports, and checkpoint/source hashes. Copy the small reports and representative images into `docs/recorded-run`; publish the large trusted checkpoint as a release/Kaggle asset with SHA-256. Do not overwrite the archived `run_kaggle-v1.py` or relabel protocol-v1 metrics.
