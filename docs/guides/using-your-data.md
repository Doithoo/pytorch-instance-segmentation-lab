# Using Your Data

Prefer the built-in COCO path when possible. Put images and train/valid/test instance JSON files under one data root, then run `instance-segment prepare-coco`; see the [dataset format reference](../reference/dataset-format.md). It supports polygons, RLE, multiclass labels, crowd flags, and empty images.

For indexed PNG masks, keep 0 for background and assign every object a distinct positive integer. Penn-Fudan is the concrete built-in example.

Use `data.factory=module:callable` only when neither format fits. Follow `examples/extensions/my_dataset.py` and return float32 CHW images plus the exact `InstanceTarget` dtypes. The factory is trusted Python code. Its manifest directory must still contain `dataset.yaml` and hashed train/valid/test CSV files so label ownership, checkpoint validation, and run comparison remain reproducible.

Before a long run, verify source hashes, inspect every split, exercise empty and multiclass samples, force horizontal flip and resize, and run a real one-batch model update.
