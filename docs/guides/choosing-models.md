# Choosing a Model

[中文](choosing-models.zh-CN.md) | [Documentation index](../README.md)

Start with the model that isolates the question you are trying to answer:

| Situation | Starting point | Reason |
| --- | --- | --- |
| Validate data, transforms, or a CPU workflow | `maskrcnn_resnet50_fpn` with `weights: none` and `learning_minimal.yaml` | Matches the default contract and avoids downloads |
| Reproduce the published Penn-Fudan baseline | `maskrcnn_resnet50_fpn` with `weights: coco_v1` | This is the recorded protocol-v2 path |
| Teach the backbone/anchor/ROI design | `maskrcnn_mobilenet_v3_large` | Smaller backbone with explicit custom components |
| Test the newer torchvision recipe | `maskrcnn_resnet50_fpn_v2` | Useful when the installed torchvision build supports it |

Check the installed catalog before writing a config:

```bash
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_mobilenet_v3_large
```

## Keep Comparisons Fair

When comparing models, hold the dataset identity, label schema, split hashes, seed, image-size policy, batch size, optimizer, learning rate, epoch budget, and evaluation thresholds fixed. Change only `model.name`, model-specific weights/parameters, and `run.name`. Use validation `valid_mask_map` for selection and evaluate the fixed test split after the choice is made.

The model catalog currently contains:

- `maskrcnn_resnet50_fpn`: the stable ResNet-50 FPN implementation with `none` and `coco_v1` weights.
- `maskrcnn_resnet50_fpn_v2`: the newer torchvision ResNet-50 FPN recipe, when available in the installed build.
- `maskrcnn_mobilenet_v3_large`: a lighter backbone with `none` and `imagenet_v2` policies.

Model-specific constructor values belong under `model.params`; commonly supported values are `min_size` and `max_size`. Do not assume that a weight name is portable between model families. Read [Model catalog](../reference/model-zoo.md) and `model-info` first.

A small dataset can make point estimates unstable. Record image/target counts, training time, peak memory, best epoch, and the complete evaluation protocol beside every comparison. The published run is a bounded teaching result, not a claim that one architecture generalizes to unrelated domains.
