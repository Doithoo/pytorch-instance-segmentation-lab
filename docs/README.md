# Documentation

This repository is easiest to use as a staged experiment. Start with the [tutorial](tutorial/README.md), then use the reference pages when you need an exact file, field, or output contract.

## Choose A Path

| Goal | Start here | Then read |
| --- | --- | --- |
| Run the Penn-Fudan example on CPU | [Environment](tutorial/01-environment.md) | [Data and instances](tutorial/02-data-and-instances.md), [Training](tutorial/04-training.md) |
| Understand what an instance target contains | [Basics](tutorial/00-basics.md) | [Instance target](concepts/instance-target.md), [Dataset format](reference/dataset-format.md) |
| Configure a reproducible experiment | [Configuration flow](concepts/configuration-flow.md) | [Configuration reference](reference/config-reference.md), [Experiments](guides/experiments.md) |
| Evaluate or inspect predictions | [Evaluation and inference](tutorial/05-evaluation-and-inference.md) | [Metrics](reference/metrics.md), [CLI and outputs](reference/cli-and-outputs.md) |
| Use COCO JSON or a custom dataset | [Using your data](guides/using-your-data.md) | [Dataset format](reference/dataset-format.md) |
| Select or add a model | [Model choice](guides/choosing-models.md) | [Model catalog](reference/model-zoo.md), [Adding models](guides/adding-models.md) |
| Reproduce the published GPU run | [Kaggle guide](guides/kaggle.md) | [Recorded run](recorded-run/README.md), [ADR 0002](architecture/0002-evaluation-and-splits.md) |

Chinese pages use the same filenames with the `.zh-CN.md` suffix. The repository also includes a root-level `mkdocs.yml` for publishing these pages as a browsable site.

## Workflow

```text
download -> prepare -> verify -> inspect -> dry-run -> train -> evaluate -> compare/predict
```

Each stage has a separate purpose:

1. Download only the source archive and verify its checksum.
2. Prepare deterministic train/valid/test manifests.
3. Verify source files and manifest hashes before an expensive run.
4. Inspect a split and render a visual preview.
5. Perform a real forward, loss, backward, and optimizer update on CPU.
6. Train while selecting `best.pt` on validation `mask_map`; test is not used by the trainer.
7. Evaluate the selected checkpoint once, compare compatible runs, or predict one image.

## Stable Contracts

- Images are float32 `CHW` tensors in `[0, 1]`.
- Targets contain aligned `boxes`, `labels`, `masks`, `image_id`, `area`, and `iscrowd` tensors.
- Label `0` is background; foreground class IDs are contiguous and start at `1`.
- Manifests and their SHA-256 identity are part of the checkpoint and evaluation contract.
- AP keeps the model's confidence ranking. `evaluation_score_floor` and the display `score_threshold` are independent.
- `.pt` checkpoints are trusted-code inputs. Verify their SHA-256 and load only trusted files.

## Repository Areas

```text
configs/        runnable YAML templates and their field guide
examples/       small executable demonstrations
scripts/        download, preview, plotting, and Kaggle build tasks
src/            package implementation in data-flow order
tests/          synthetic and integration tests
docs/           tutorials, concepts, guides, references, ADRs, and run evidence
```

The [recorded run](recorded-run/README.md) is evidence for one bounded Penn-Fudan protocol, not a general benchmark. The [model card](recorded-run/MODEL_CARD.md) describes its intended use and limitations.
