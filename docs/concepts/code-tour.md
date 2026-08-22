# Code Tour

[中文](code-tour.zh-CN.md) | [Documentation index](../README.md)

Read the implementation in data-flow order. The CLI is intentionally thin: it parses arguments and delegates work to testable package APIs.

1. `src/instance_segmenter/config.py` defines dataclass defaults, strict YAML merging, `--set` overrides, and validation.
2. `src/instance_segmenter/data/manifest.py` defines dataset metadata, split hashes, and Penn-Fudan preparation/verification.
3. `src/instance_segmenter/data/coco.py` decodes polygon and RLE annotations and builds the COCO provider.
4. `src/instance_segmenter/data/schema.py` defines `InstanceTarget`, `LabelSchema`, and dtype/shape validation.
5. `src/instance_segmenter/data/transforms.py` keeps image, mask, box, and area geometry aligned.
6. `src/instance_segmenter/data/collate.py` preserves variable image sizes and variable instance counts as lists.
7. `src/instance_segmenter/models/registry.py` and `models/torchvision_models.py` expose the model catalog and predictor replacement.
8. `src/instance_segmenter/training/trainer.py` performs one epoch or the real dry-run update; `training/train.py` owns the reproducible loop.
9. `src/instance_segmenter/training/checkpoint.py` validates and restores trusted training state.
10. `src/instance_segmenter/evaluation/metrics.py` computes COCO-style bbox/mask metrics; `evaluation/evaluate.py` writes reports and ranked overlays.
11. `src/instance_segmenter/inference/predictor.py` turns one checkpoint and image into JSON, binary masks, and an overlay.
12. `src/instance_segmenter/cli.py` connects these APIs to `instance-segment` commands.

## Extension Boundaries

A dataset provider must return `(image, target)` pairs that satisfy the target contract and must be addressable by the manifest metadata. A model factory must return a `torch.nn.Module` with the torchvision detection convention:

- training: `model(images, targets)` returns a finite loss dictionary;
- evaluation: `model(images)` returns one prediction dictionary per image;
- prediction fields: `boxes`, `labels`, `scores`, and `masks` stay aligned on their first dimension.

Use the examples under `examples/extensions/` as the smallest extension starting point. Keep trusted factory loading explicit with `module.path:callable`; do not hide arbitrary imports in the CLI.

## Artifact Boundaries

`src/` contains reusable behavior. `scripts/` contains repository and publication tasks. `examples/` demonstrates contracts without becoming a second trainer. `docs/recorded-run/` contains evidence and provenance, not runtime data used by the package. Preserve these boundaries when adding a feature so documentation, tests, and the generated Kaggle runner remain auditable.
