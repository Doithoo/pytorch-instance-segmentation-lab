# Configurations

- `learning_minimal.yaml`: tiny CPU dry-run and local smoke configuration. It limits train/valid/test to a few rows and uses `weights: none`.
- `maskrcnn_resnet50_fpn.yaml`: untrained ResNet50-FPN template for larger local experiments.
- `maskrcnn_mobilenet_v3_large.yaml`: lightweight ImageNet-backbone model configuration.
- `reference_maskrcnn.yaml`: protocol-v2 20-epoch Kaggle GPU replacement baseline with all split rows and `coco_v1` weights.
- `custom_dataset_example.yaml`: trusted external dataset factory contract.
- `custom_model_example.yaml`: trusted external model factory contract.

Copy an installed template with `instance-segment init-config NAME --output config.yaml`. Use `instance-segment show-config --config config.yaml` to inspect resolved values and their sources. Metric AP uses `evaluation_score_floor`, while prediction/error display uses `score_threshold`; do not conflate them.
