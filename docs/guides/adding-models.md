# Adding a Model

[中文](adding-models.zh-CN.md) | [Documentation index](../README.md)

A model can be built in the registry or supplied as a trusted external factory. Both paths must implement the same torchvision detection contract.

## Contract

The factory receives `num_classes`, `weights`, and `params`. Training calls `model(images, targets)` and expects a dictionary of finite scalar losses. Evaluation calls `model(images)` and expects one dictionary per image with aligned `boxes`, `labels`, `scores`, and `masks`. Masks must remain independent instances; do not merge touching objects into one semantic mask.

## Built-in Model Checklist

1. Implement the factory under `src/instance_segmenter/models/` and keep constructor parameters explicit.
2. Add a `ModelSpec` in `models/registry.py` with a stable name, description, supported weight policies, input notes, and parameter notes.
3. Replace both box and mask predictors when `num_classes` differs from the upstream default.
4. Add tests for supported/unsupported weights, predictor head shapes, empty predictions, a real training forward, and checkpoint restoration.
5. Add or update a runnable config and the model catalog in both languages.
6. Run the CPU smoke path and, if the model claims GPU support, run a real one-batch CUDA smoke test.

## External Factory

Create a module importable from the repository environment and set:

```yaml
model:
  factory: my_project.models:build_segmenter
  name: external-segmenter
  weights: none
  num_classes: 2
  params:
    min_size: 128
```

The callable should accept keyword arguments and return `torch.nn.Module`. External factories execute Python code during training, evaluation, and inference, so they are trusted-code inputs. Keep the factory close to its tests and document its weight downloads and license terms.

Do not add a model name without documenting its predictor heads, weight policy, image-size assumptions, and supported torchvision versions. A dependency and registry entry alone does not establish a usable model contract.
