# 配置

- `learning_minimal.yaml`：小型 CPU dry-run 与本地 smoke 配置，train/valid/test 只取少量样本并使用 `weights: none`。
- `maskrcnn_resnet50_fpn.yaml`：用于更大规模本地实验的未预训练 ResNet50-FPN 模板。
- `maskrcnn_mobilenet_v3_large.yaml`：使用 ImageNet backbone 的轻量模型配置。
- `reference_maskrcnn.yaml`：协议 v2 的 20 epoch Kaggle GPU 替代基线，使用全部 split 行和 `coco_v1` 权重。
- `custom_dataset_example.yaml`：可信外部 dataset factory 契约。
- `custom_model_example.yaml`：可信外部 model factory 契约。

使用 `instance-segment init-config NAME --output config.yaml` 复制安装后的模板，使用 `instance-segment show-config --config config.yaml` 查看解析值及其来源。指标 AP 使用 `evaluation_score_floor`，推理/错误展示使用 `score_threshold`，两者不要混用。
