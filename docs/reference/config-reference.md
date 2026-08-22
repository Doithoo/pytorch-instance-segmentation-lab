# Configuration Reference

[中文](config-reference.zh-CN.md) | [Documentation index](../README.md)

The loader accepts YAML fields that correspond to `AppConfig`. Unknown fields are rejected. Paths are interpreted relative to the current working directory.

| Section | Fields | Defaults / notes |
| --- | --- | --- |
| `run` | `name`, `seed`, `output_dir` | `learning-minimal`, `42`, `artifacts`; run path is `output_dir/name` |
| `data` | `provider`, `factory`, `root`, `manifest_dir` | Provider is `pennfudan` or `coco`; factory is trusted `module:callable` |
| `data` | `image_size`, `batch_size`, `num_workers` | Default resize `[128,128]`, batch `1`, workers `0`; `null` preserves source size |
| `data` | `horizontal_flip`, `train_limit`, `valid_limit`, `test_limit` | Flip probability `0.0`; limits are positive or `null` for all rows |
| `model` | `name`, `factory`, `weights`, `num_classes`, `params` | Two classes including background; params go to the factory |
| `training` | `epochs`, `optimizer`, `learning_rate`, `momentum`, `weight_decay` | SGD, 2 epochs, `0.005`, `0.9`, `0.0005` |
| `training` | `scheduler`, `step_size`, `gamma` | `none`, `6`, `0.1`; scheduler can be `step` |
| `training` | `amp`, `grad_clip_norm`, `evaluate_every` | `auto`, no clipping, evaluate every epoch |
| `training` | `best_metric` | Fixed to `mask_map` |
| `training` | `evaluation_score_floor` | `0.0`; optional input floor for AP |
| `training` | `score_threshold`, `mask_threshold` | `0.5`, `0.5`; display/error and binary-mask thresholds |
| root | `device` | `auto`; choices are `auto`, `cpu`, `cuda`, `mps` |

## Model Parameters

`model.params` is the only open mapping. Built-in factories commonly accept `min_size` and `max_size`; use `instance-segment model-info NAME` for model-specific notes. Weight names are validated by the selected model, so `coco_v1` is not interchangeable with `imagenet_v2`.

## CLI Overrides

Use one `--set KEY VALUE` pair per override. Values are parsed with YAML, so `false`, `null`, numbers, lists, and strings retain their types:

```bash
uv run instance-segment show-config --config configs/learning_minimal.yaml \
  --set data.image_size null --set training.amp true --set run.name full-size
```

The CLI override has highest precedence. `show-config` also reports the source of each resolved leaf. `train --device` overrides only the selected runtime device.

## Reproducibility Fields

For comparisons, keep `run.seed`, provider, data root contents, manifest identity, label schema, model name/weights/params, image-size policy, loader/augmentation settings, optimization values, and all thresholds explicit. The trainer persists the resolved config and split hashes; checkpoint resume rejects incompatible immutable fields.
