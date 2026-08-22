# ADR-0001: PyTorch Instance Segmentation Lab

Status: planned. The Chinese document is the canonical implementation specification:

[`0001-instance-segmentation-lab.zh-CN.md`](0001-instance-segmentation-lab.zh-CN.md)

The project is a separate repository from both `pytorch-object-detection-lab`
and `pytorch-image-segmentation-lab`. Its first learning path is Penn-Fudan
Pedestrian plus torchvision Mask R-CNN, with a reproducible manifest, synthetic
CPU dry-run, checkpointing, COCO-style bbox/mask metrics, and single-image
instance visualization. Full reference training is a required Kaggle GPU path:
a generated runner embeds the exact source snapshot, completes all 20 epochs on
a T4 or newer compatible GPU, selects the best checkpoint on validation mask AP,
and evaluates the test split once after model selection.

The implementation should follow the Chinese specification phase by phase:

1. initialize packaging and CLI;
2. implement the instance target contract and synthetic tests;
3. implement Penn-Fudan download, parsing, inspection, and manifests;
4. add model registry and Mask R-CNN factories;
5. add training, checkpoints, evaluation, and inference;
6. add tutorials and extension examples;
7. generate and validate the self-contained Kaggle runner, complete the full GPU
   reference run, and publish auditable recorded-run artifacts without the large
   checkpoint.
