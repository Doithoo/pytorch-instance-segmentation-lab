# Using Models

Use `instance-segment list-models` and `model-info`. An external model factory receives `num_classes`, `weights`, and `params`, returns `nn.Module`, and must match torchvision's training/inference contract. `examples/extensions/my_segmenter.py` delegates to the built-in Mask R-CNN factory.
