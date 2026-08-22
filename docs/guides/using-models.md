# Using Models

Run `instance-segment list-models` and `model-info NAME` before selecting a configuration. ResNet50-FPN v1 is the replacement reference path; v2 exposes the newer torchvision recipe; MobileNetV3-Large demonstrates explicit custom anchors and ROI pooling with a lighter backbone.

Weight policies are model-specific. `coco_v1` loads complete Mask R-CNN COCO weights before replacing both prediction heads. `imagenet_v2` initializes only the MobileNet backbone. `none` avoids downloads.

An external model factory uses `module:callable`, receives keyword-only `num_classes`, `weights`, and `params`, and returns an `nn.Module` that follows the torchvision training/evaluation contract. External factories execute trusted Python code. Validate head dimensions, finite component losses, output field alignment, empty predictions, and checkpoint restoration before full training.
