# Configuration Flow

[中文](configuration-flow.zh-CN.md) | [Documentation index](../README.md)

Configuration is deliberately strict so a typo cannot silently change an experiment.

## Precedence

The loader starts from dataclass defaults, merges a YAML file, then applies each repeated `--set KEY VALUE` override:

```text
default dataclasses < YAML < CLI --set
```

`--device` on `train` is a command-line replacement for the top-level device field. `show-config` prints the resolved YAML plus a `sources` map showing whether each leaf came from the default, YAML, or CLI.

```bash
uv run instance-segment show-config --config configs/learning_minimal.yaml \
  --set training.epochs 3 --set model.params.min_size 160
```

YAML and overrides accept only known top-level and section fields. `model.params` is the intentional escape hatch for model constructor parameters; it is still passed to a trusted model factory.

## What Is Captured

A full run stores the resolved config in `config.yaml` and embeds it in both checkpoints. It also records the manifest split hashes, environment, source state, and lockfile hash. This means a result can be audited without guessing which defaults were active.

## Important Sections

- `run`: `name`, `seed`, and `output_dir`; the run directory is `output_dir/name`.
- `data`: provider/factory, data root, manifest directory, image resize, loader settings, augmentation probability, and split limits.
- `model`: registered name or trusted factory, weight policy, class count, and constructor params.
- `training`: optimizer/scheduler, loss optimization values, AMP, clipping, validation cadence, and thresholds.
- `device`: `auto`, `cpu`, `cuda`, or `mps`.

`training.best_metric` is intentionally fixed to `mask_map`. `training.evaluation_score_floor` controls which predictions enter AP; `training.score_threshold` controls readable analysis and prediction display; `training.mask_threshold` binarizes masks. They should not be substituted for one another.

See the [configuration reference](../reference/config-reference.md) for defaults and validation rules.
