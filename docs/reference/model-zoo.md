# Model Zoo

| Name | Weight policies | Intended use |
|---|---|---|
| `maskrcnn_resnet50_fpn` | `none`, `coco_v1` | Stable transfer-learning and full reference runner |
| `maskrcnn_resnet50_fpn_v2` | `none`, `coco_v1` | Newer torchvision ResNet50-FPN recipe |
| `maskrcnn_mobilenet_v3_large` | `none`, `imagenet_v2` | Lightweight Mask R-CNN with an inspectable custom backbone |

COCO Mask R-CNN policies load complete pretrained weights and replace box and mask predictors for the configured class count. The MobileNet variant uses an ImageNet backbone, one feature map, custom anchors, and explicit box/mask ROI pooling; it does not claim COCO-pretrained instance heads.

`model.params` is forwarded to the corresponding constructor, commonly for `min_size` and `max_size`. Run `instance-segment model-info NAME` for model-specific notes. Any newly advertised model should have head-shape tests, a real forward smoke test in its supported environment, and a documented weight policy.
