# Configurations

- `learning_minimal.yaml`: tiny CPU dry-run and local smoke configuration.
- `maskrcnn_resnet50_fpn.yaml`: full untrained ResNet50-FPN template.
- `maskrcnn_mobilenet_v3_large.yaml`: lightweight ImageNet-backbone model configuration.
- `reference_maskrcnn.yaml`: protocol-v2 20-epoch Kaggle GPU replacement baseline configuration.
- `custom_dataset_example.yaml`: trusted external dataset factory contract.
- `custom_model_example.yaml`: trusted external model factory contract.

Copy an installed template with `instance-segment init-config NAME --output config.yaml`. Metric AP uses `evaluation_score_floor`, while prediction/error display uses `score_threshold`; do not conflate them.
