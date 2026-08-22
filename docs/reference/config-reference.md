# Configuration Reference

`run.name`, `seed`, and `output_dir` identify an artifact directory. `data.provider` is `pennfudan` or `coco`; `data.factory=module:callable` remains available for trusted plugins. Root, manifest directory, resize, loader, augmentation, and split limits are explicit.

`model.name`, weight policy, class count, and constructor `params` define the model. Use `instance-segment model-info NAME` before forwarding torchvision parameters.

Training config includes optimizer/scheduler values, AMP policy, gradient clipping, evaluation cadence, and three distinct thresholds:

- `evaluation_score_floor`: optional lower bound before confidence-ranked AP, normally `0.0`.
- `score_threshold`: prediction display and per-image error threshold.
- `mask_threshold`: mask probability binarization threshold.

`training.best_metric` is fixed to `mask_map`. The final epoch is always evaluated even when it is outside `evaluate_every`. `device` is `auto`, `cpu`, `cuda`, or `mps`.

YAML accepts only known fields. `--set section.field VALUE` has highest precedence. `instance-segment show-config` shows resolved values and their source; `init-config` copies templates from source checkouts or installed wheels.
